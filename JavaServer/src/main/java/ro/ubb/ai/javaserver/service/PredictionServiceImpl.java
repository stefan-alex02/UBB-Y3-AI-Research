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
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class PredictionServiceImpl implements PredictionService {

    private final PredictionRepository predictionRepository;
    private final ImageRepository imageRepository;
    private final ExperimentRepository experimentRepository;
    private final UserRepository userRepository;
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

        List<ImagePredictionTaskDTO> tasksForPython = new ArrayList<>();
        List<Prediction> newPredictionEntities = new ArrayList<>();
        List<Prediction> oldPredictionsToDeleteArtifactsFor = new ArrayList<>();

        for (Long imageIdFromRequest : createRequest.getImageIds()) {
            Image image = imageRepository.findById(imageIdFromRequest)
                    .orElseThrow(() -> new ResourceNotFoundException("Image", "id", imageIdFromRequest));
            if (!image.getUser().getId().equals(user.getId())) {
                log.warn("User {} not authorized for image {}, skipping in batch.", username, imageIdFromRequest);
                continue;
            }

            predictionRepository.findByImageIdAndModelExperimentExperimentRunId(image.getId(), modelExperiment.getExperimentRunId())
                    .ifPresent(oldPrediction -> {
                        oldPredictionsToDeleteArtifactsFor.add(oldPrediction);
                        predictionRepository.delete(oldPrediction);
                        predictionRepository.flush();
                        log.info("Removed old SQL prediction record: id {}, image {}, model_exp {}",
                                oldPrediction.getId(), image.getId(), modelExperiment.getExperimentRunId());
                    });


            Prediction newPredictionShell = new Prediction();
            newPredictionShell.setImage(image);
            newPredictionShell.setModelExperiment(modelExperiment);
            newPredictionShell.setPredictedClass("PENDING");
            newPredictionShell.setConfidence(0.0f);
            Prediction savedNewPrediction = predictionRepository.saveAndFlush(newPredictionShell);

            newPredictionEntities.add(savedNewPrediction);

            ImagePredictionTaskDTO task = new ImagePredictionTaskDTO();
            task.setImageId(String.valueOf(image.getId()));
            task.setImageFormat(image.getFormat());
            task.setPredictionId(String.valueOf(savedNewPrediction.getId()));
            tasksForPython.add(task);
        }

        if (tasksForPython.isEmpty()) {
            throw new IllegalArgumentException("No valid images for prediction in the batch.");
        }

        // Prepare and Call Python
        PythonPredictionRequestDTO pythonRequest = new PythonPredictionRequestDTO();
        pythonRequest.setUsername(username);
        pythonRequest.setImagePredictionTasks(tasksForPython);

        ModelLoadDetailsDTO modelDetails = new ModelLoadDetailsDTO();
        modelDetails.setDatasetNameOfModel(modelExperiment.getDatasetName());
        modelDetails.setModelTypeOfModel(modelExperiment.getModelType());
        modelDetails.setExperimentRunIdOfModelProducer(modelExperiment.getExperimentRunId());
        modelDetails.setRelativeModelPathInExperiment(modelExperiment.getModelRelativePath());

        pythonRequest.setModelLoadDetails(modelDetails);
        pythonRequest.setExperimentRunIdOfModel(modelExperiment.getExperimentRunId());
        pythonRequest.setGenerateLime(createRequest.getGenerateLime());
        pythonRequest.setLimeNumFeatures(createRequest.getLimeNumFeatures());
        pythonRequest.setLimeNumSamples(createRequest.getLimeNumSamples());
        pythonRequest.setProbPlotTopK(createRequest.getProbPlotTopK());

        PythonPredictionRunResponseDTO pythonResponse;
        try {
            pythonResponse = pythonApiService.runPredictionInPython(pythonRequest);
        } catch (Exception e) {
            log.error("Python prediction call failed: {}", e.getMessage());
            throw new RuntimeException("Prediction processing failed in Python backend.", e);
        }

        // Update new Prediction entities
        List<PredictionDTO> createdPredictionDTOs = new ArrayList<>();
        if (pythonResponse == null || pythonResponse.getPredictions() == null) {
            throw new RuntimeException("Python service returned null or empty predictions response.");
        }

        for (PythonSinglePredictionResultDTO pyPredResult : pythonResponse.getPredictions()) {
            Long currentPredictionId = Long.valueOf(pyPredResult.getPredictionId());

            newPredictionEntities.stream()
                    .filter(p -> p.getId().equals(currentPredictionId))
                    .findFirst()
                    .ifPresentOrElse(p -> {
                        p.setPredictedClass(pyPredResult.getPredictedClass());
                        p.setConfidence(pyPredResult.getConfidence());
                        createdPredictionDTOs.add(convertToDTO(p));
                    }, () -> log.warn("No matching Prediction entity found for ID: {}", currentPredictionId));
        }

        // Delete the artifacts of the OLD predictions
        for (Prediction oldPrediction : oldPredictionsToDeleteArtifactsFor) {
            try {
                log.info("Deleting old artifacts for prediction ID: {}", oldPrediction.getId());
                pythonApiService.deletePythonPredictionArtifacts(
                        username,
                        String.valueOf(oldPrediction.getImage().getId()),
                        String.valueOf(oldPrediction.getId())
                );
            } catch (Exception e) {
                log.error("Failed to delete old artifacts for prediction ID {}: {}. New prediction still created.", oldPrediction.getId(), e.getMessage());
            }
        }

        if (createdPredictionDTOs.size() != tasksForPython.size()) {
            log.warn("Mismatch in number of tasks sent to Python and results processed. Sent: {}, Processed DTOs: {}", tasksForPython.size(), createdPredictionDTOs.size());
        }

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
    public PredictionDTO getPrediction(Long predictionId, String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));
        Prediction prediction = predictionRepository.findById(predictionId)
                .orElseThrow(() -> new ResourceNotFoundException("Prediction", "id", predictionId));
        if (!prediction.getImage().getUser().getId().equals(user.getId())) {
            throw new SecurityException("User not authorized to view this prediction.");
        }
        return convertToDTO(prediction);
    }


    @Override
    @Transactional
    public void deletePrediction(Long predictionId, String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));
        Prediction prediction = predictionRepository.findById(predictionId)
                .orElseThrow(() -> new ResourceNotFoundException("Prediction", "id", predictionId));
        if (!prediction.getImage().getUser().getId().equals(user.getId())) {
            throw new SecurityException("User not authorized to delete this prediction.");
        }

        pythonApiService.deletePythonPredictionArtifacts(
                username,
                String.valueOf(prediction.getImage().getId()),
                String.valueOf(prediction.getId())
        );
        predictionRepository.delete(prediction);
        log.info("Prediction ID {} deleted from DB and artifacts.", predictionId);
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
            dto.setModelExperimentRunId(null);
            dto.setModelExperimentName("Unknown Exp.");
        }
        dto.setPredictedClass(prediction.getPredictedClass());
        dto.setConfidence(prediction.getConfidence());
        dto.setPredictionTimestamp(prediction.getPredictionTimestamp());
        return dto;
    }
}
