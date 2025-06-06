package ro.ubb.ai.javaserver.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ro.ubb.ai.javaserver.dto.prediction.*;
import ro.ubb.ai.javaserver.entity.Experiment;
import ro.ubb.ai.javaserver.entity.Image;
import ro.ubb.ai.javaserver.entity.Prediction;
import ro.ubb.ai.javaserver.entity.User;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;
import ro.ubb.ai.javaserver.repository.ExperimentRepository;
import ro.ubb.ai.javaserver.repository.ImageRepository;
import ro.ubb.ai.javaserver.repository.PredictionRepository;
import ro.ubb.ai.javaserver.repository.UserRepository;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class PredictionServiceImpl implements PredictionService {

    private final PredictionRepository predictionRepository;
    private final ImageRepository imageRepository;
    private final ExperimentRepository experimentRepository;
    private final UserRepository userRepository; // For username access
    private final PythonApiService pythonApiService;

    @Override
    @Transactional
    public PredictionDTO createPrediction(PredictionCreateRequest createRequest, String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));

        Image image = imageRepository.findById(createRequest.getImageId())
                .orElseThrow(() -> new ResourceNotFoundException("Image", "id", createRequest.getImageId()));

        if (!image.getUser().getId().equals(user.getId())) {
            throw new SecurityException("User not authorized to create prediction for this image.");
        }

        Experiment modelExperiment = experimentRepository.findById(createRequest.getModelExperimentRunId())
                .orElseThrow(() -> new ResourceNotFoundException("Experiment (for model)", "id", createRequest.getModelExperimentRunId()));

        if (modelExperiment.getModelRelativePath() == null || modelExperiment.getModelRelativePath().isBlank()) {
            throw new IllegalArgumentException("Selected experiment " + modelExperiment.getExperimentRunId() + " does not have a saved model path.");
        }

        // Prepare request for Python
        PythonPredictionRequestDTO pythonRequest = new PythonPredictionRequestDTO();
        pythonRequest.setUsername(username);

        ImageIdFormatPairDTO imagePair = new ImageIdFormatPairDTO();
        imagePair.setImageId(String.valueOf(image.getId())); // Python expects string for image_id in this DTO
        imagePair.setImageFormat(image.getFormat());
        pythonRequest.setImageIdFormatPairs(Collections.singletonList(imagePair));

        ModelLoadDetailsDTO modelDetails = new ModelLoadDetailsDTO();
        modelDetails.setDatasetNameOfModel(modelExperiment.getDatasetName());
        modelDetails.setModelTypeOfModel(modelExperiment.getModelType());
        modelDetails.setExperimentRunIdOfModelProducer(modelExperiment.getExperimentRunId());
        modelDetails.setRelativeModelPathInExperiment(modelExperiment.getModelRelativePath());
        pythonRequest.setModelLoadDetails(modelDetails);

        pythonRequest.setExperimentRunIdOfModel(modelExperiment.getExperimentRunId()); // For grouping results
        pythonRequest.setGenerateLime(createRequest.getGenerateLime());
        pythonRequest.setLimeNumFeatures(createRequest.getLimeNumFeatures());
        pythonRequest.setLimeNumSamples(createRequest.getLimeNumSamples());
        pythonRequest.setProbPlotTopK(createRequest.getProbPlotTopK());

        // Call Python API
        PythonPredictionRunResponseDTO pythonResponse = pythonApiService.runPredictionInPython(pythonRequest);

        if (pythonResponse == null || pythonResponse.getPredictions() == null || pythonResponse.getPredictions().isEmpty()) {
            log.error("Python prediction service returned no predictions for imageId: {}", image.getId());
            throw new RuntimeException("Prediction failed: Python service returned no results.");
        }

        // Assuming single image prediction for now, take the first result
        PythonSinglePredictionResultDTO pyPredResult = pythonResponse.getPredictions().get(0);

        // Check if a prediction for this image with this model already exists
        predictionRepository.findByImageIdAndModelExperimentExperimentRunId(image.getId(), modelExperiment.getExperimentRunId())
                .ifPresent(existingPrediction -> {
                    log.info("Deleting existing prediction for image {} with model from experiment {}", image.getId(), modelExperiment.getExperimentRunId());
                    predictionRepository.delete(existingPrediction);
                    // TODO: Optionally call Python API to delete old prediction artifacts from MinIO
                });

        Prediction newPrediction = new Prediction();
        newPrediction.setImage(image);
        newPrediction.setModelExperiment(modelExperiment);
        newPrediction.setPredictedClass(pyPredResult.getPredictedClass());
        newPrediction.setConfidence(pyPredResult.getConfidence());

        Prediction savedPrediction = predictionRepository.save(newPrediction);
        log.info("Prediction saved for imageId: {}, model from experiment: {}", image.getId(), modelExperiment.getExperimentRunId());
        return convertToDTO(savedPrediction);
    }

    @Override
    public List<PredictionDTO> getPredictionsForImage(Long imageId, String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));
        Image image = imageRepository.findById(imageId)
                .orElseThrow(() -> new ResourceNotFoundException("Image", "id", imageId));
        if (!image.getUser().getId().equals(user.getId())) {
            throw new SecurityException("User not authorized to view predictions for this image.");
        }
        return predictionRepository.findByImageIdOrderByPredictionTimestampDesc(imageId)
                .stream().map(this::convertToDTO).collect(Collectors.toList());
    }

    @Override
    public PredictionDTO getPrediction(Long imageId, String modelExperimentRunId, String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));
        Image image = imageRepository.findById(imageId)
                .orElseThrow(() -> new ResourceNotFoundException("Image", "id", imageId));
        if (!image.getUser().getId().equals(user.getId())) {
            throw new SecurityException("User not authorized to view this prediction.");
        }
        Prediction prediction = predictionRepository.findByImageIdAndModelExperimentExperimentRunId(imageId, modelExperimentRunId)
                .orElseThrow(() -> new ResourceNotFoundException("Prediction", "image_id and model_experiment_id", imageId + "/" + modelExperimentRunId));
        return convertToDTO(prediction);
    }


    @Override
    @Transactional
    public void deletePrediction(Long imageId, String modelExperimentRunId, String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));
        Image image = imageRepository.findById(imageId)
                .orElseThrow(() -> new ResourceNotFoundException("Image", "id", imageId));
        if (!image.getUser().getId().equals(user.getId())) {
            throw new SecurityException("User not authorized to delete this prediction.");
        }

        // TODO: Call Python API to delete prediction artifacts from MinIO
        // pythonApiService.deletePythonPredictionArtifacts(username, String.valueOf(imageId), modelExperimentRunId);
        log.warn("Python API call for deleting prediction artifacts for image {}, model {} from MinIO is not yet implemented.", imageId, modelExperimentRunId);


        predictionRepository.deleteByImageIdAndModelExperimentExperimentRunId(imageId, modelExperimentRunId);
        log.info("Prediction for imageId: {}, model from experiment: {} deleted from DB.", imageId, modelExperimentRunId);
    }

    private PredictionDTO convertToDTO(Prediction prediction) {
        PredictionDTO dto = new PredictionDTO();
        dto.setId(prediction.getId());
        dto.setImageId(prediction.getImage().getId());
        dto.setModelExperimentRunId(prediction.getModelExperiment().getExperimentRunId());
        dto.setPredictedClass(prediction.getPredictedClass());
        dto.setConfidence(prediction.getConfidence());
        dto.setPredictionTimestamp(prediction.getPredictionTimestamp());
        return dto;
    }
}
