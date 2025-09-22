package ro.ubb.ai.javaserver.dto.prediction;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Data
public class PredictionCreateRequest {
    @NotNull
    private List<Long> imageIds;

    @NotBlank
    private String modelExperimentRunId;

    private Boolean generateLime;
    private Integer limeNumFeatures;
    private Integer limeNumSamples;
    private Integer probPlotTopK;
}
