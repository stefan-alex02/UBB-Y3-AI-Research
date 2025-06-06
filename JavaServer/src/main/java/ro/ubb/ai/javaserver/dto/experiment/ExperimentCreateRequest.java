package ro.ubb.ai.javaserver.dto.experiment;

import lombok.Data;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;

import java.util.List;

@Data
public class ExperimentCreateRequest {
    @NotBlank
    @Size(max = 255)
    private String name; // User-friendly name

    @NotBlank
    @Size(max = 50)
    private String modelType;

    @NotBlank
    @Size(max = 100)
    private String datasetName;

    @NotEmpty
    private List<PythonExperimentMethodParamsDTO> methodsSequence;

    // Optional overrides for Python PipelineExecutor defaults
    private Integer imgSizeH;
    private Integer imgSizeW;
    private Boolean saveModelDefault;
    private Boolean offlineAugmentation;
    private String augmentationStrategyOverride;
    // Add other executor-level params Java should control
}

