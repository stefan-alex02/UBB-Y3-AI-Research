package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;

@Data
public class ModelLoadDetailsDTO { // Corresponds to Python's ModelLoadDetails
    private String datasetNameOfModel;
    private String modelTypeOfModel;
    private String experimentRunIdOfModelProducer;
    private String relativeModelPathInExperiment;
}
