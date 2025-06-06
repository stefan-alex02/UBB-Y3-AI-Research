package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;
import java.time.OffsetDateTime;

@Data
public class PredictionDTO {
    private Long id;
    private Long imageId;
    private String modelExperimentRunId; // ID of the experiment that produced the model
    private String predictedClass;
    private Float confidence;
    private OffsetDateTime predictionTimestamp;
    // You might add fields for artifact paths (LIME, prob plot) to be constructed on frontend
}
