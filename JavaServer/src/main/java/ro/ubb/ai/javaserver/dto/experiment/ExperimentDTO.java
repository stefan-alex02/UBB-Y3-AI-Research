package ro.ubb.ai.javaserver.dto.experiment;

import lombok.Data;
import ro.ubb.ai.javaserver.enums.ExperimentStatus;

import java.time.OffsetDateTime;
import java.util.List;

@Data
public class ExperimentDTO {
    private String experimentRunId;
    private Long userId;
    private String userName;
    private String name;
    private String modelType;
    private String datasetName;
    private String modelRelativePath;
    private ExperimentStatus status;
    private OffsetDateTime startTime;
    private OffsetDateTime endTime;
    private List<PythonExperimentMethodParamsDTO> sequenceConfig;
}
