package ro.ubb.ai.javaserver.dto.experiment;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.List;

@Data
public class ExperimentCreateRequest {
    @NotBlank
    @Size(max = 255)
    private String name;

    @NotBlank
    @Size(max = 50)
    private String modelType;

    @NotBlank
    @Size(max = 100)
    private String datasetName;

    @NotEmpty
    private List<PythonExperimentMethodParamsDTO> methodsSequence;

    private Integer imgSizeH;
    private Integer imgSizeW;
    private Boolean offlineAugmentation;
    private String augmentationStrategyOverride;
    private Float testSplitRatioIfFlat;
    private Integer randomSeed;
    private Boolean forceFlatForFixedCv;
}

