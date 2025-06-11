package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;
import java.util.List;

@Data
public class PythonPredictionRequestDTO {
    private String username;
    private List<ImagePredictionTaskDTO> imagePredictionTasks;
    private ModelLoadDetailsDTO modelLoadDetails;
    private String experimentRunIdOfModel;

    private Boolean generateLime;
    private Integer limeNumFeatures;
    private Integer limeNumSamples;
    private Integer probPlotTopK;
}

