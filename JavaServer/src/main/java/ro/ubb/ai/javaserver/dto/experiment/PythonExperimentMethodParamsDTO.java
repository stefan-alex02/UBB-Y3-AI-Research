package ro.ubb.ai.javaserver.dto.experiment;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashMap;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PythonExperimentMethodParamsDTO {

    @NotBlank
    private String methodName;

    private Map<String, Object> params = new HashMap<>();
    private Map<String, Object> paramGrid;

    private Boolean saveModel;
    private Boolean saveBestModel;
    private Integer plotLevel;
    private Integer resultsDetailLevel;
    private Integer cv;
    private Integer outerCv;
    private Integer innerCv;
    private String scoring;
    private String methodSearchType;
    @JsonProperty("n_iter")
    private Integer nIter;
    private String evaluateOn;
    private Float valSplitRatio;
    private Integer useBestParamsFromStep;
}
