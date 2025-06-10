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

import java.util.ArrayList;
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
    public List<PredictionDTO> createPrediction(PredictionCreateRequest createRequest, String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));

        Experiment modelExperiment = experimentRepository.findById(createRequest.getModelExperimentRunId())
                .orElseThrow(() -> new ResourceNotFoundException("Experiment (for model)", "id", createRequest.getModelExperimentRunId()));

        if (modelExperiment.getModelRelativePath() == null || modelExperiment.getModelRelativePath().isBlank()) {
            throw new IllegalArgumentException("Selected experiment " + modelExperiment.getExperimentRunId() + " does not have a saved model path.");
        }

        List<ImageIdFormatPairDTO> imagePairsForPython = new ArrayList<>();
        List<Image> validImagesForDb = new ArrayList<>();

        for (Long imageId : createRequest.getImageIds()) {
            Image image = imageRepository.findById(imageId)
                    .orElseThrow(() -> new ResourceNotFoundException("Image", "id", imageId));
            if (!image.getUser().getId().equals(user.getId())) {
                log.warn("User {} not authorized for image {}, skipping.", username, imageId);
                continue; // Or throw an error for the whole batch
            }
            ImageIdFormatPairDTO pair = new ImageIdFormatPairDTO();
            pair.setImageId(String.valueOf(imageId));
            pair.setImageFormat(image.getFormat());
            imagePairsForPython.add(pair);
            validImagesForDb.add(image); // Keep track of valid Image entities
        }

        if (imagePairsForPython.isEmpty()) {
            throw new IllegalArgumentException("No valid or authorized images found for prediction.");
        }

        // Prepare request for Python
        PythonPredictionRequestDTO pythonRequest = new PythonPredictionRequestDTO();
        pythonRequest.setUsername(username);
        pythonRequest.setImageIdFormatPairs(imagePairsForPython);

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

        if (pythonResponse == null || pythonResponse.getPredictions() == null ||
                pythonResponse.getPredictions().size() != validImagesForDb.size()) { // Expect one result per valid image sent
            log.error("Python prediction service returned inconsistent results. Expected: {}, Got: {}",
                    validImagesForDb.size(), pythonResponse.getPredictions() != null ? pythonResponse.getPredictions().size() : 0);
            throw new RuntimeException("Prediction failed: Python service returned inconsistent results.");
        }

        List<PredictionDTO> createdPredictionDTOs = new ArrayList<>();
        for (int i = 0; i < validImagesForDb.size(); i++) {
            Image image = validImagesForDb.get(i);
            PythonSinglePredictionResultDTO pyPredResult = pythonResponse.getPredictions().stream()
                    .filter(p -> String.valueOf(image.getId()).equals(p.getImageId()))
                    .findFirst()
                    .orElse(null);

            if (pyPredResult == null) {
                log.warn("No prediction result from Python for imageId: {}. Skipping DB save for this one.", image.getId());
                continue;
            }

            // **Handle Deletion of Old Prediction Artifacts First**
            if (modelExperiment != null) { // Only if a specific model is used
                predictionRepository.findByImageIdAndModelExperimentExperimentRunId(image.getId(), modelExperiment.getExperimentRunId())
                        .ifPresent(existingPrediction -> {
                            log.info("Overwriting existing prediction for image {} with model from experiment {}. Deleting old artifacts first.",
                                    image.getId(), modelExperiment.getExperimentRunId());
                            // Call Python to delete old prediction artifacts for this specific imageId + modelExperimentRunId
                            try {
                                pythonApiService.deletePythonPredictionArtifacts(
                                        username,
                                        String.valueOf(image.getId()),
                                        modelExperiment.getExperimentRunId()
                                );
                            } catch (Exception e) {
                                log.error("Failed to delete old prediction artifacts for image {}, model_exp {}: {}",
                                        image.getId(), modelExperiment.getExperimentRunId(), e.getMessage());
                                // Decide if this should stop the process or just log and continue
                            }
                            predictionRepository.delete(existingPrediction); // Delete old DB record
                        });
            }


            Prediction newPrediction = new Prediction();
            newPrediction.setImage(image);
            newPrediction.setModelExperiment(modelExperiment);
            newPrediction.setPredictedClass(pyPredResult.getPredictedClass());
            newPrediction.setConfidence(pyPredResult.getConfidence());
            Prediction savedPrediction = predictionRepository.save(newPrediction);
            createdPredictionDTOs.add(convertToDTO(savedPrediction));
        }
        log.info("Batch prediction processed for {} images for user {}.", createdPredictionDTOs.size(), username);
        return createdPredictionDTOs;
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
        imageRepository.findById(imageId).orElseThrow(() -> new ResourceNotFoundException("Image", "id", imageId));
        // Find the prediction to ensure it exists and user has rights via image ownership
        Prediction prediction = predictionRepository.findByImageIdAndModelExperimentExperimentRunId(imageId, modelExperimentRunId)
                .orElseThrow(() -> new ResourceNotFoundException("Prediction", "imageId and modelExperimentRunId", imageId + "/" + modelExperimentRunId));
        if (!prediction.getImage().getUser().getId().equals(user.getId())) {
            throw new SecurityException("User not authorized to delete this prediction.");
        }

        // Call Python to delete prediction artifacts folder from MinIO
        pythonApiService.deletePythonPredictionArtifacts(
                username, String.valueOf(imageId), modelExperimentRunId
        );

        predictionRepository.delete(prediction); // Delete from DB
        log.info("Prediction for imageId: {}, model from experiment: {} deleted.", imageId, modelExperimentRunId);
    }

    private PredictionDTO convertToDTO(Prediction prediction) {
        PredictionDTO dto = new PredictionDTO();
        dto.setId(prediction.getId());
        dto.setImageId(prediction.getImage().getId());
        if (prediction.getModelExperiment() != null) {
            dto.setModelExperimentRunId(prediction.getModelExperiment().getExperimentRunId());
            dto.setModelExperimentName(prediction.getModelExperiment().getName());
            dto.setModelType(prediction.getModelExperiment().getModelType());
            dto.setDatasetName(prediction.getModelExperiment().getDatasetName());

        } else {
            dto.setModelExperimentRunId(null); // Or a placeholder like "UNKNOWN_EXPERIMENT"
            dto.setModelExperimentName("Unknown Exp.");
        }
        dto.setPredictedClass(prediction.getPredictedClass());
        dto.setConfidence(prediction.getConfidence());
        dto.setPredictionTimestamp(prediction.getPredictionTimestamp());
        return dto;
    }
}
