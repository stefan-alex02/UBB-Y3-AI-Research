package ro.ubb.ai.javaserver.controller;

import ro.ubb.ai.javaserver.dto.experiment.ExperimentCreateRequest;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentDTO;
import ro.ubb.ai.javaserver.dto.experiment.ExperimentFilterDTO;
import ro.ubb.ai.javaserver.service.ExperimentService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/experiments")
@RequiredArgsConstructor
public class ExperimentController {

    private final ExperimentService experimentService;

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
    @PreAuthorize("hasRole('METEOROLOGIST')")
    public ResponseEntity<Page<ExperimentDTO>> getAllExperiments(
            @ModelAttribute ExperimentFilterDTO filterDTO, // Use @ModelAttribute for query params
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "startTime") String sortBy,
            @RequestParam(defaultValue = "DESC") String sortDir) {

        Sort.Direction direction = Sort.Direction.fromString(sortDir.toUpperCase());
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortBy));
        Page<ExperimentDTO> experiments = experimentService.filterExperiments(filterDTO, pageable);
        return ResponseEntity.ok(experiments);
    }

    // TODO DELETE endpoint for experiments
}
