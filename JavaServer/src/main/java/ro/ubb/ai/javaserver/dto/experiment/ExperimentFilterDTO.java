package ro.ubb.ai.javaserver.dto.experiment;

import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;
import ro.ubb.ai.javaserver.enums.ExperimentStatus;

import java.time.OffsetDateTime;

@Data
public class ExperimentFilterDTO {
    private String nameContains;
    private String modelType;
    private String datasetName;
    private ExperimentStatus status;
    private Boolean hasModelSaved;
    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
    private OffsetDateTime startedAfter;
    @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
    private OffsetDateTime finishedBefore;
}
