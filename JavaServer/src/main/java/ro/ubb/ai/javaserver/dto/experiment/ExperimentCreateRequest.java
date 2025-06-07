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
    private String name; // User-friendly name - THIS IS NOW THE PRIMARY NAME

    @NotBlank
    @Size(max = 50)
    private String modelType;

    @NotBlank
    @Size(max = 100)
    private String datasetName;

    @NotEmpty
    private List<PythonExperimentMethodParamsDTO> methodsSequence; // This DTO has Map<String, Object> params

    // Optional overrides
    private Integer imgSizeH;
    private Integer imgSizeW;
    // private Boolean saveModelDefault; // Removed from React UI, can be removed here too if Python executor handles it
    private Boolean offlineAugmentation;
    private String augmentationStrategyOverride;
}

