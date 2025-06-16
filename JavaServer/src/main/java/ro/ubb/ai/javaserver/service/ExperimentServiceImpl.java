package ro.ubb.ai.javaserver.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import ro.ubb.ai.javaserver.dto.experiment.*;
import ro.ubb.ai.javaserver.entity.Experiment;
import ro.ubb.ai.javaserver.entity.User;
import ro.ubb.ai.javaserver.enums.ExperimentStatus;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;
import ro.ubb.ai.javaserver.repository.ExperimentRepository;
import ro.ubb.ai.javaserver.repository.UserRepository;
import ro.ubb.ai.javaserver.repository.specification.ExperimentSpecification;
import ro.ubb.ai.javaserver.websocket.ExperimentStatusWebSocketHandler;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class ExperimentServiceImpl implements ExperimentService {

    private final ExperimentRepository experimentRepository;
    private final UserRepository userRepository;
    private final PythonApiService pythonApiService;
    private final ObjectMapper objectMapper;
    private final ExperimentStatusWebSocketHandler webSocketHandler;

    @Override
    @Transactional
    public ExperimentDTO createExperiment(ExperimentCreateRequest createRequest) {
        String currentUsername = SecurityContextHolder.getContext().getAuthentication().getName();
        User currentUser = userRepository.findByUsername(currentUsername)
                .orElseThrow(() -> new RuntimeException("User not found for experiment creation: " + currentUsername));

        Experiment experiment = new Experiment();
        String systemGeneratedRunId = UUID.randomUUID().toString();
        experiment.setExperimentRunId(systemGeneratedRunId);
        experiment.setUser(currentUser);
        experiment.setName(createRequest.getName());
        experiment.setModelType(createRequest.getModelType());
        experiment.setDatasetName(createRequest.getDatasetName());
        experiment.setStatus(ExperimentStatus.PENDING);

        try {
            String sequenceConfigJson = objectMapper.writeValueAsString(createRequest.getMethodsSequence());
            experiment.setSequenceConfig(sequenceConfigJson);
        } catch (JsonProcessingException e) {
            log.error("Error serializing sequence_config for experiment {}: {}", systemGeneratedRunId, e.getMessage());
            throw new RuntimeException("Failed to prepare experiment configuration.", e);
        }

        Experiment savedExperiment = experimentRepository.save(experiment);
        log.info("Experiment '{}' (ID: {}) saved to DB with PENDING status by user {}.",
                savedExperiment.getName(), systemGeneratedRunId, currentUsername);

        // Prepare DTO for Python API
        PythonRunExperimentRequestDTO pythonRequest = new PythonRunExperimentRequestDTO(
                systemGeneratedRunId,
                createRequest.getDatasetName(),
                createRequest.getModelType(),
                createRequest.getMethodsSequence(),
                createRequest.getImgSizeH(),
                createRequest.getImgSizeW(),
                null,
                createRequest.getOfflineAugmentation(),
                createRequest.getAugmentationStrategyOverride(),
                createRequest.getTestSplitRatioIfFlat(),
                createRequest.getRandomSeed(),
                createRequest.getForceFlatForFixedCv()
        );

        try {
            pythonApiService.startPythonExperiment(pythonRequest);
            log.info("Request to start Python experiment ID {} (Name: '{}') sent.", systemGeneratedRunId, createRequest.getName());
        } catch (Exception e) {
            log.error("Failed to send start_experiment request to Python for ID {}: {}. Marking as FAILED.", systemGeneratedRunId, e.getMessage());
            savedExperiment.setStatus(ExperimentStatus.FAILED);
            savedExperiment.setEndTime(OffsetDateTime.now());
            experimentRepository.save(savedExperiment);
            throw new RuntimeException("Failed to initiate experiment with Python service: " + e.getMessage(), e);
        }

        return convertToDTO(savedExperiment);
    }

    @Override
    @Transactional(readOnly = true)
    public ExperimentDTO getExperimentById(String experimentRunId) {
        Experiment experiment = experimentRepository.findById(experimentRunId)
                .orElseThrow(() -> new ResourceNotFoundException("Experiment", "experimentRunId", experimentRunId));
        return convertToDTO(experiment);
    }

    @Override
    @Transactional(readOnly = true)
    public Page<ExperimentDTO> filterExperiments(ExperimentFilterDTO filterDTO, Pageable pageable) {
        log.debug("Filtering experiments with DTO: {} and Pageable: {}", filterDTO, pageable);
        Specification<Experiment> spec = ExperimentSpecification.fromFilter(filterDTO);
        Page<Experiment> experimentsPage = experimentRepository.findAll(spec, pageable);
        log.info("Found {} experiments matching filter.", experimentsPage.getTotalElements());
        return experimentsPage.map(this::convertToDTO);
    }

    @Override
    @Transactional
    public void deleteExperiment(String experimentRunId) {
        Experiment experiment = experimentRepository.findById(experimentRunId)
                .orElseThrow(() -> new ResourceNotFoundException("Experiment", "experimentRunId", experimentRunId));

        pythonApiService.deletePythonExperimentArtifacts(
                experiment.getDatasetName(), experiment.getModelType(), experimentRunId
        );

        experimentRepository.delete(experiment);
        log.info("Experiment with ID {} deleted from DB. Associated predictions updated.", experimentRunId);
    }

    @Override
    @Transactional
    public ExperimentDTO updateExperiment(String experimentRunId, ExperimentStatus status, String modelRelativePath, Boolean setEndTime, String errorMessage) {
        Experiment experiment = experimentRepository.findById(experimentRunId)
                .orElseThrow(() -> new ResourceNotFoundException("Experiment not found: " + experimentRunId));

        experiment.setStatus(status);
        if (modelRelativePath != null && !modelRelativePath.isBlank()) {
            experiment.setModelRelativePath(modelRelativePath);
        }
        if (Boolean.TRUE.equals(setEndTime) || status == ExperimentStatus.COMPLETED || status == ExperimentStatus.FAILED) {
            if (experiment.getEndTime() == null) {
                experiment.setEndTime(OffsetDateTime.now());
            }
        }

        Experiment updatedExperiment = experimentRepository.save(experiment);
        log.info("Experiment {} updated in DB. Status: {}, ModelPath: {}, Error: {}",
                experimentRunId, status, modelRelativePath != null ? "Set" : "Not Set", errorMessage != null ? "Yes" : "No");

        ExperimentDTO dto = convertToDTO(updatedExperiment);

        webSocketHandler.broadcastExperimentUpdate(dto);

        return dto;
    }

    private ExperimentDTO convertToDTO(Experiment experiment) {
        ExperimentDTO dto = new ExperimentDTO();
        dto.setExperimentRunId(experiment.getExperimentRunId());
        if (experiment.getUser() != null) {
            dto.setUserId(experiment.getUser().getId());
            dto.setUserName(experiment.getUser().getUsername());
        }
        dto.setName(experiment.getName());
        dto.setModelType(experiment.getModelType());
        dto.setDatasetName(experiment.getDatasetName());
        dto.setModelRelativePath(experiment.getModelRelativePath());
        dto.setStatus(experiment.getStatus());
        dto.setStartTime(experiment.getStartTime());
        dto.setEndTime(experiment.getEndTime());

        if (experiment.getSequenceConfig() != null && !experiment.getSequenceConfig().isBlank()) {
            try {
                List<PythonExperimentMethodParamsDTO> sequence = objectMapper.readValue(
                        experiment.getSequenceConfig(),
                        new TypeReference<List<PythonExperimentMethodParamsDTO>>() {}
                );
                dto.setSequenceConfig(sequence);
            } catch (JsonProcessingException e) {
                log.error("Error deserializing sequence_config for experiment {}: {}", experiment.getExperimentRunId(), e.getMessage());
                dto.setSequenceConfig(null);
            }
        } else {
            dto.setSequenceConfig(null);
        }
        return dto;
    }
}
