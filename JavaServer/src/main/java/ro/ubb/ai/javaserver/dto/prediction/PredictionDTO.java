package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;

import java.time.OffsetDateTime;

@Data
public class PredictionDTO {
    private Long id;
    private Long imageId;
    private String modelExperimentRunId;
    private String modelExperimentName;
    private String modelType;
    private String datasetName;
    private String predictedClass;
    private Float confidence;
    private OffsetDateTime predictionTimestamp;
}
