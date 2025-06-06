package ro.ubb.ai.javaserver.dto.experiment;

import ro.ubb.ai.javaserver.enums.ExperimentStatus;
import lombok.Data;

@Data
public class ExperimentUpdateRequest {
    private ExperimentStatus status;
    private String modelRelativePath; // Python provides this when model is saved
    private Boolean setEndTime; // If true, endTime will be set to now
    private String errorMessage; // Optional, if status is FAILED
}