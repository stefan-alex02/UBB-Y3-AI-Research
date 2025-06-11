package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;

@Data
public class ImagePredictionTaskDTO {
    private String imageId;
    private String imageFormat;
    private String predictionId;
}
