package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;

@Data
public class PythonSinglePredictionResultDTO { // Corresponds to Python's SinglePredictionResult
    private String imageId;
    private String predictionId;
    private String experimentId;
    private String predictedClass;
    private Float confidence;
}
