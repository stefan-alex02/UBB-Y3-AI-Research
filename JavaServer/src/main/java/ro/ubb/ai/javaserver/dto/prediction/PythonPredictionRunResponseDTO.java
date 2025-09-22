package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;

import java.util.List;

@Data
public class PythonPredictionRunResponseDTO {
    private List<PythonSinglePredictionResultDTO> predictions;
    private String message;
}

