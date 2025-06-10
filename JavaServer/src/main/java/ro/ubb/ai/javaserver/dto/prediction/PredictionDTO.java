package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;
import java.time.OffsetDateTime;

@Data
public class PredictionDTO {
    private Long id; // Assuming Integer PK
    private Long imageId;
    private String modelExperimentRunId; // Can be null
    private String modelExperimentName; // New field for display
    private String modelType;
    private String datasetName;
    private String predictedClass;
    private Float confidence;
    private OffsetDateTime predictionTimestamp;
}
