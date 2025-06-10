package ro.ubb.ai.javaserver.dto.experiment;

import ro.ubb.ai.javaserver.enums.ExperimentStatus;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;
import java.time.OffsetDateTime;

@Data
public class ExperimentFilterDTO {
    private String nameContains; // New: Filter by experiment name (user-defined name)
    private String modelType;
    private String datasetName;
    private ExperimentStatus status;
    private Boolean hasModelSaved; // New: Filter by whether model_relative_path is set
    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
    private OffsetDateTime startedAfter;
    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
    private OffsetDateTime finishedBefore;
    // Pagination params are handled by Pageable in controller
}
