package ro.ubb.ai.javaserver.dto.experiment;

import com.fasterxml.jackson.annotation.JsonProperty;
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
    private Map<String, Object> paramGrid; // Specifically for search methods' HP space

    // Add all optional method-specific arguments that React sends
    // These correspond to direct arguments for Python pipeline methods,
    // NOT nested inside the 'params' map above.
    private Boolean saveModel;
    private Boolean saveBestModel;
    private Integer plotLevel;
    private Integer resultsDetailLevel;
    private Integer cv; // For non_nested_grid_search (inner cv) and cv_model_evaluation (k-folds)
    private Integer outerCv; // For nested_grid_search
    private Integer innerCv; // For nested_grid_search
    private String scoring;
    private String methodSearchType; // "grid" or "random" (for search methods)
    @JsonProperty("n_iter")
    private Integer nIter;
    private String evaluateOn;
    private Float valSplitRatio; // Standardized name
    private Integer useBestParamsFromStep;

    // If you have other parameters sent at this level from React, add them here.
    // For example, if 'use_best_params_from_step_checkbox' was being sent (though it shouldn't be,
    // only the derived 'use_best_params_from_step' integer value should be sent).
}
