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
    private Map<String, Object> params = new HashMap<>();
}
