package ro.ubb.ai.javaserver.service;

import ro.ubb.ai.javaserver.dto.prediction.PredictionCreateRequest;
import ro.ubb.ai.javaserver.dto.prediction.PredictionDTO;

import java.util.List;

public interface PredictionService {
    List<PredictionDTO> createPrediction(PredictionCreateRequest createRequest, String username);
    List<PredictionDTO> getPredictionsForImage(Long imageId, String username);
    PredictionDTO getPrediction(Long predictionId, String username);
    void deletePrediction(Long predictionId, String username);
}
