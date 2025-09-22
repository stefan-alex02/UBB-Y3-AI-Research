package ro.ubb.ai.javaserver.internal_api;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentDTO;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentUpdateRequest;
import ro.ubb.ai.javaserver.service.ExperimentService;

@RestController
@RequestMapping("/api/internal/experiments")
@RequiredArgsConstructor
@Slf4j
public class InternalExperimentController {

    private final ExperimentService experimentService;

    @Value("${python.api.internal-key}")
    private String expectedInternalApiKey;

    private static final String INTERNAL_API_KEY_HEADER = "X-Internal-API-Key";


    @PutMapping("/{experimentRunId}/update")
    public ResponseEntity<ExperimentDTO> updateExperimentStatus(
            @PathVariable String experimentRunId,
            @RequestBody ExperimentUpdateRequest updateRequest,
            @RequestHeader(value = INTERNAL_API_KEY_HEADER, required = true) String apiKey) {

        log.info("Received internal request to update experiment {}: Status={}, ModelPath provided: {}",
                experimentRunId, updateRequest.getStatus(), updateRequest.getModelRelativePath() != null);

        if (!expectedInternalApiKey.equals(apiKey)) {
            log.warn("Invalid internal API key received for experiment update: {}", experimentRunId);
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        try {
            ExperimentDTO updatedExperiment = experimentService.updateExperiment(
                    experimentRunId,
                    updateRequest.getStatus(),
                    updateRequest.getModelRelativePath(),
                    updateRequest.getSetEndTime(),
                    updateRequest.getErrorMessage()
            );
            return ResponseEntity.ok(updatedExperiment);
        } catch (RuntimeException e) {
            log.error("Error updating experiment {} internally: {}", experimentRunId, e.getMessage());
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
    }
}
