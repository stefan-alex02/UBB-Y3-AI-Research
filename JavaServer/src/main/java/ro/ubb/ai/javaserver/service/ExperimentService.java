package ro.ubb.ai.javaserver.service;

import ro.ubb.ai.javaserver.dto.experiment.ExperimentCreateRequest;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentDTO;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentFilterDTO;
import ro.ubb.ai.javaserver.enums.ExperimentStatus;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

public interface ExperimentService {
    ExperimentDTO createExperiment(ExperimentCreateRequest createRequest);
    ExperimentDTO getExperimentById(String experimentRunId);
    Page<ExperimentDTO> filterExperiments(ExperimentFilterDTO filterDTO, Pageable pageable);
    void deleteExperiment(String experimentRunId);
    ExperimentDTO updateExperiment(String experimentRunId, ExperimentStatus status, String modelRelativePath, Boolean setEndTime, String errorMessage);
}
