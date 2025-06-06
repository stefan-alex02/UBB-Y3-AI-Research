package ro.ubb.ai.javaserver.dto.experiment;

import ro.ubb.ai.javaserver.enums.ExperimentStatus;
import lombok.Data;
import java.time.OffsetDateTime;
import java.util.List; // For sequence config display
import java.util.Map;

@Data
public class ExperimentDTO {
    private String experimentRunId;
    private Long userId;
    private String userName; // User who initiated
    private String name;
    private String modelType;
    private String datasetName;
    private String modelRelativePath;
    private ExperimentStatus status;
    private OffsetDateTime startTime;
    private OffsetDateTime endTime;
    private List<PythonExperimentMethodParamsDTO> sequenceConfig; // Deserialized from JSON string
}
