package ro.ubb.ai.javaserver.dto.experiment;

import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PythonRunExperimentRequestDTO {
    private String experimentRunId;
    private String datasetName;
    private String modelType;
    private List<PythonExperimentMethodParamsDTO> methodsSequence;
    private Integer imgSizeH;
    private Integer imgSizeW;
    private Boolean saveModelDefault;
    private Boolean offlineAugmentation;
    private String augmentationStrategyOverride;
}
