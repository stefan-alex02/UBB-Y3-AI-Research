package ro.ubb.ai.javaserver.dto.experiment;

import lombok.Data;
import ro.ubb.ai.javaserver.enums.ExperimentStatus;

@Data
public class ExperimentUpdateRequest {
    private ExperimentStatus status;
    private String modelRelativePath;
    private Boolean setEndTime;
    private String errorMessage;
}