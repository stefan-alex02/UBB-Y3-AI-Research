package ro.ubb.ai.javaserver.controller;

import ro.ubb.ai.javaserver.dto.prediction.PredictionCreateRequest;
import ro.ubb.ai.javaserver.dto.prediction.PredictionDTO;
import ro.ubb.ai.javaserver.service.PredictionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/predictions")
@RequiredArgsConstructor
public class PredictionController {

    private final PredictionService predictionService;

    @PostMapping
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<PredictionDTO> createPrediction(@Valid @RequestBody PredictionCreateRequest createRequest) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        PredictionDTO prediction = predictionService.createPrediction(createRequest, currentUsername);
        return new ResponseEntity<>(prediction, HttpStatus.CREATED);
    }

    @GetMapping("/image/{imageId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<List<PredictionDTO>> getPredictionsForImage(@PathVariable Long imageId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        return ResponseEntity.ok(predictionService.getPredictionsForImage(imageId, currentUsername));
    }

    @GetMapping("/image/{imageId}/model/{modelExperimentRunId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<PredictionDTO> getSpecificPrediction(
            @PathVariable Long imageId,
            @PathVariable String modelExperimentRunId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        return ResponseEntity.ok(predictionService.getPrediction(imageId, modelExperimentRunId, currentUsername));
    }


    @DeleteMapping("/image/{imageId}/model/{modelExperimentRunId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Void> deletePrediction(
            @PathVariable Long imageId,
            @PathVariable String modelExperimentRunId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        predictionService.deletePrediction(imageId, modelExperimentRunId, currentUsername);
        return ResponseEntity.noContent().build();
    }
}
