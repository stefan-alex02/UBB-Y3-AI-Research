package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;
import java.util.List;

@Data
public class PythonPredictionRequestDTO {
    private String username;
    private List<ImageIdFormatPairDTO> imageIdFormatPairs; // List of images to predict
    private ModelLoadDetailsDTO modelLoadDetails;
    private String experimentRunIdOfModel; // For grouping results under the model's experiment ID

    private Boolean generateLime;
    private Integer limeNumFeatures;
    private Integer limeNumSamples;
    private Integer probPlotTopK;
}

