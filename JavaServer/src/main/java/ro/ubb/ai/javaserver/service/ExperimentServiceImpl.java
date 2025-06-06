package ro.ubb.ai.javaserver.service;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentCreateRequest;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentDTO;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentFilterDTO;
import ro.ubb.ai.javaserver.dto.experiment.PythonRunExperimentRequestDTO;
import ro.ubb.ai.javaserver.entity.Experiment;
import ro.ubb.ai.javaserver.entity.User;
import ro.ubb.ai.javaserver.enums.ExperimentStatus;
import ro.ubb.ai.javaserver.repository.ExperimentRepository;
import ro.ubb.ai.javaserver.repository.UserRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.OffsetDateTime;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class ExperimentServiceImpl implements ExperimentService {

    private final ExperimentRepository experimentRepository;
    private final UserRepository userRepository;
    private final PythonApiService pythonApiService;
    private final ObjectMapper objectMapper; // For serializing sequenceConfig

    @Override
    @Transactional
    public ExperimentDTO createExperiment(ExperimentCreateRequest createRequest) {
        String currentUsername = SecurityContextHolder.getContext().getAuthentication().getName();
        User currentUser = userRepository.findByUsername(currentUsername)
                .orElseThrow(() -> new RuntimeException("User not found for experiment creation"));

        Experiment experiment = new Experiment();
        String experimentRunId = UUID.randomUUID().toString(); // Generate ID in Java
        experiment.setExperimentRunId(experimentRunId);
        experiment.setUser(currentUser);
        experiment.setName(createRequest.getName());
        experiment.setModelType(createRequest.getModelType());
        experiment.setDatasetName(createRequest.getDatasetName());
        experiment.setStatus(ExperimentStatus.PENDING);
        experiment.setStartTime(OffsetDateTime.now());

        try {
            // Serialize the methodsSequence to JSON string for sequence_config
            String sequenceConfigJson = objectMapper.writeValueAsString(createRequest.getMethodsSequence());
            experiment.setSequenceConfig(sequenceConfigJson);
        } catch (JsonProcessingException e) {
            log.error("Error serializing sequence_config for experiment {}: {}", experimentRunId, e.getMessage());
            throw new RuntimeException("Failed to prepare experiment configuration.", e);
        }

        Experiment savedExperiment = experimentRepository.save(experiment);
        log.info("Experiment {} saved to DB with PENDING status.", experimentRunId);

        // Prepare DTO for Python API
        PythonRunExperimentRequestDTO pythonRequest = new PythonRunExperimentRequestDTO(
                experimentRunId,
                createRequest.getDatasetName(),
                createRequest.getModelType(),
                createRequest.getMethodsSequence(), // Pass the List<PythonExperimentMethodParamsDTO>
                createRequest.getImgSizeH(),
                createRequest.getImgSizeW(),
                createRequest.getSaveModelDefault(),
                createRequest.getOfflineAugmentation(),
                createRequest.getAugmentationStrategyOverride()
        );

        // Call Python API (this will be asynchronous in the sense that Python runs it in background)
        try {
            pythonApiService.startPythonExperiment(pythonRequest);
            log.info("Request to start experiment {} sent to Python service.", experimentRunId);
        } catch (Exception e) {
            log.error("Failed to send start_experiment request to Python for {}: {}. Marking as FAILED.", experimentRunId, e.getMessage());
            // If sending the request to Python fails, mark the experiment as FAILED in DB
            savedExperiment.setStatus(ExperimentStatus.FAILED);
            savedExperiment.setEndTime(OffsetDateTime.now());
            experimentRepository.save(savedExperiment);
            // Re-throw or handle appropriately for the frontend
            throw new RuntimeException("Failed to initiate experiment with Python service.", e);
        }

        return convertToDTO(savedExperiment);
    }

    @Override
    public ExperimentDTO getExperimentById(String experimentRunId) {
        return null;
    }

    @Override
    public Page<ExperimentDTO> filterExperiments(ExperimentFilterDTO filterDTO, Pageable pageable) {
        return null;
    }

    @Override
    public void deleteExperiment(String experimentRunId) {

    }

    @Override
    @Transactional
    public ExperimentDTO updateExperiment(String experimentRunId, ExperimentStatus status, String modelRelativePath, Boolean setEndTime, String errorMessage) {
        Experiment experiment = experimentRepository.findById(experimentRunId)
                .orElseThrow(() -> new RuntimeException("Experiment not found: " + experimentRunId));

        experiment.setStatus(status);
        if (modelRelativePath != null && !modelRelativePath.isBlank()) {
            experiment.setModelRelativePath(modelRelativePath);
        }
        if (Boolean.TRUE.equals(setEndTime) || status == ExperimentStatus.COMPLETED || status == ExperimentStatus.FAILED) {
            experiment.setEndTime(OffsetDateTime.now());
        }
        // Could add error message to a new field in Experiment entity if status is FAILED

        Experiment updatedExperiment = experimentRepository.save(experiment);
        log.info("Experiment {} updated in DB. Status: {}, ModelPath: {}",
                experimentRunId, status, modelRelativePath != null ? "Set" : "Not Set");
        return convertToDTO(updatedExperiment);
    }

    // Other methods: getExperimentById, filterExperiments, deleteExperiment...

    private ExperimentDTO convertToDTO(Experiment experiment) {
        // Use a DTO that includes necessary fields for the frontend
        ExperimentDTO dto = new ExperimentDTO();
        dto.setExperimentRunId(experiment.getExperimentRunId());
        dto.setUserId(experiment.getUser().getId());
        dto.setUserName(experiment.getUser().getUsername()); // Or name
        dto.setName(experiment.getName());
        dto.setModelType(experiment.getModelType());
        dto.setDatasetName(experiment.getDatasetName());
        dto.setModelRelativePath(experiment.getModelRelativePath());
        dto.setStatus(experiment.getStatus());
        dto.setStartTime(experiment.getStartTime());
        dto.setEndTime(experiment.getEndTime());
        // You might not send the full sequenceConfig JSON to the list view,
        // but maybe to the detail view.
        // dto.setSequenceConfig(experiment.getSequenceConfig());
        return dto;
    }
}
