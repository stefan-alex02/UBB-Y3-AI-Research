package ro.ubb.ai.javaserver.dto.experiment;

import ro.ubb.ai.javaserver.enums.ExperimentStatus;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;
import java.time.OffsetDateTime;

@Data
public class ExperimentFilterDTO {
    private String modelType;
    private String datasetName;
    private ExperimentStatus status;
    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
    private OffsetDateTime startedAfter;
    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
    private OffsetDateTime finishedBefore;
    // Add pagination params: page, size, sortBy, sortDir
    private Integer page;
    private Integer size;
    private String sortBy;
    private String sortDir;
}
