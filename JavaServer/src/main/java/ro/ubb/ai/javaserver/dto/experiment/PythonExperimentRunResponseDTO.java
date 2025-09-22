package ro.ubb.ai.javaserver.dto.experiment;

import lombok.Data;

@Data
public class PythonExperimentRunResponseDTO {
    private String experimentRunId;
    private String message;
    private String status;
}
