package ro.ubb.ai.javaserver.dto.experiment;

import lombok.Data;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import jakarta.validation.constraints.NotBlank;
import java.util.Map;
import java.util.HashMap;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PythonExperimentMethodParamsDTO {

    @NotBlank
    private String methodName;

    private Map<String, Object> params = new HashMap<>(); // For Skorch HPs or GridSearchCV config

    // Add all optional method-specific arguments that React sends
    // These correspond to direct arguments for Python pipeline methods,
    // NOT nested inside the 'params' map above.
    private Boolean saveModel; // Matches 'save_model' from React
    private Boolean saveBestModel; // Matches 'save_best_model' from React
    private Integer plotLevel;
    private Integer resultsDetailLevel;
    private Integer cv;
    private Integer outerCv;
    private Integer innerCv;
    private String scoring;
    private String methodSearchType; // e.g., "grid" or "random" (React sends this)
    // Python side will map this to 'method' for PipelineExecutor
    private Integer nIter;
    private String evaluateOn;
    private Float valSplitRatio;
    private Integer useBestParamsFromStep;

    // If you have other parameters sent at this level from React, add them here.
    // For example, if 'use_best_params_from_step_checkbox' was being sent (though it shouldn't be,
    // only the derived 'use_best_params_from_step' integer value should be sent).
}
