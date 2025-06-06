package ro.ubb.ai.javaserver.dto.prediction;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Data;
import java.util.List;

@Data
public class PredictionCreateRequest {
    @NotNull
    private Long imageId; // The image we are predicting on

    @NotBlank
    private String modelExperimentRunId; // The ExperimentRunId of the model to use

    // Python specific params
    private Boolean generateLime;
    private Integer limeNumFeatures;
    private Integer limeNumSamples;
    private Integer probPlotTopK;
}
