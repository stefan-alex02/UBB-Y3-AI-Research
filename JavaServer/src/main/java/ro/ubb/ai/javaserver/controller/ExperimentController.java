package ro.ubb.ai.javaserver.controller;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentCreateRequest;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentDTO;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentFilterDTO;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;
import ro.ubb.ai.javaserver.service.ExperimentService;
import ro.ubb.ai.javaserver.service.PythonApiService;

import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/experiments")
@RequiredArgsConstructor
public class ExperimentController {

    private final ExperimentService experimentService;
    private final PythonApiService pythonApiService;

    @PostMapping
    @PreAuthorize("hasRole('METEOROLOGIST')")
    public ResponseEntity<ExperimentDTO> createExperiment(@Valid @RequestBody ExperimentCreateRequest createRequest) {
        ExperimentDTO experiment = experimentService.createExperiment(createRequest);
        return new ResponseEntity<>(experiment, HttpStatus.CREATED);
    }

    @GetMapping("/{experimentRunId}")
    @PreAuthorize("hasRole('METEOROLOGIST')")
    public ResponseEntity<ExperimentDTO> getExperiment(@PathVariable String experimentRunId) {
        return ResponseEntity.ok(experimentService.getExperimentById(experimentRunId));
    }

    @GetMapping
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Page<ExperimentDTO>> getAllExperiments(
            @ModelAttribute ExperimentFilterDTO filterDTO,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "startTime") String sortBy,
            @RequestParam(defaultValue = "DESC") String sortDir) {

        Sort.Direction direction = "ASC".equalsIgnoreCase(sortDir) ? Sort.Direction.ASC : Sort.Direction.DESC;
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortBy));
        Page<ExperimentDTO> experiments = experimentService.filterExperiments(filterDTO, pageable);
        return ResponseEntity.ok(experiments);
    }

    @DeleteMapping("/{experimentRunId}")
    @PreAuthorize("hasRole('METEOROLOGIST')")
    public ResponseEntity<Void> deleteExperiment(@PathVariable String experimentRunId) {
        experimentService.deleteExperiment(experimentRunId);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/{experimentRunId}/artifacts/list")
    @PreAuthorize("hasRole('METEOROLOGIST')")
    public ResponseEntity<List<Map<String, Object>>> listArtifacts(
            @PathVariable String experimentRunId,
            @RequestParam(required = false, defaultValue = "") String path) {
        ExperimentDTO experiment = experimentService.getExperimentById(experimentRunId);
        if (experiment == null) {
            return ResponseEntity.notFound().build();
        }
        List<Map<String, Object>> artifacts = pythonApiService.listPythonExperimentArtifacts(
                experiment.getDatasetName(), experiment.getModelType(), experimentRunId, path
        );
        return ResponseEntity.ok(artifacts);
    }

    @GetMapping("/{experimentRunId}/artifacts/content/**")
    @PreAuthorize("hasRole('METEOROLOGIST')")
    public ResponseEntity<Resource> getArtifactContent(
            @PathVariable String experimentRunId,
            HttpServletRequest request
    ) {
        ExperimentDTO experiment = experimentService.getExperimentById(experimentRunId);
        if (experiment == null) {
            return ResponseEntity.notFound().build();
        }

        String fullRequestPath = request.getRequestURI();
        String basePathToTrim = String.format("/api/experiments/%s/artifacts/content/", experimentRunId);
        String artifactRelativePath = fullRequestPath.substring(basePathToTrim.length());

        if (artifactRelativePath.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }

        try {
            byte[] contentBytes = pythonApiService.getPythonExperimentArtifactContent(
                    experiment.getDatasetName(), experiment.getModelType(), experimentRunId, artifactRelativePath
            );

            HttpHeaders headers = new HttpHeaders();

            String filename;
            try {
                filename = Paths.get(artifactRelativePath).getFileName().toString();
            } catch (Exception e) {
                log.warn("Could not parse filename using Paths.get for: '{}', falling back. Error: {}", artifactRelativePath, e.getMessage());
                int lastSlash = artifactRelativePath.lastIndexOf('/');
                filename = (lastSlash >= 0) ? artifactRelativePath.substring(lastSlash + 1) : artifactRelativePath;
            }

            String mediaType = "application/octet-stream";
            if (filename.toLowerCase().endsWith(".json")) mediaType = MediaType.APPLICATION_JSON_VALUE;
            else if (filename.toLowerCase().endsWith(".png")) mediaType = MediaType.IMAGE_PNG_VALUE;
            else if (filename.toLowerCase().endsWith(".jpg") || filename.toLowerCase().endsWith(".jpeg")) mediaType = MediaType.IMAGE_JPEG_VALUE;
            else if (filename.toLowerCase().endsWith(".log") || filename.toLowerCase().endsWith(".txt")) mediaType = MediaType.TEXT_PLAIN_VALUE + ";charset=UTF-8";
            else if (filename.toLowerCase().endsWith(".csv")) mediaType = "text/csv;charset=UTF-8";

            headers.setContentType(MediaType.parseMediaType(mediaType));

            ByteArrayResource resource = new ByteArrayResource(contentBytes);
            return ResponseEntity.ok().headers(headers).contentLength(contentBytes.length).body(resource);

        } catch (ResourceNotFoundException e) {
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            log.error("Error proxying artifact content for experiment {}: {}", experimentRunId, e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
}
