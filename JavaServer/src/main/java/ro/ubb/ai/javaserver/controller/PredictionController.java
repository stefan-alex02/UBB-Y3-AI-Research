package ro.ubb.ai.javaserver.controller;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.util.UrlPathHelper;
import ro.ubb.ai.javaserver.dto.prediction.PredictionCreateRequest;
import ro.ubb.ai.javaserver.dto.prediction.PredictionDTO;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;
import ro.ubb.ai.javaserver.service.PredictionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import ro.ubb.ai.javaserver.service.PythonApiService;

import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/predictions")
@RequiredArgsConstructor
public class PredictionController {

    private final PredictionService predictionService;
    private final PythonApiService pythonApiService; // Inject
    private final UrlPathHelper urlPathHelper = new UrlPathHelper();

    @PostMapping
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<List<PredictionDTO>> createPrediction(@Valid @RequestBody PredictionCreateRequest createRequest) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        List<PredictionDTO> prediction = predictionService.createPrediction(createRequest, currentUsername);
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

    @GetMapping("/{imageId}/model/{modelExperimentRunId}/artifacts/list")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<List<Map<String, Object>>> listPredictionArtifacts(
            @PathVariable Long imageId,
            @PathVariable String modelExperimentRunId,
            @RequestParam(required = false, defaultValue = "") String path,
            Authentication authentication) {
        String currentUsername = authentication.getName();
        // Optional: Add a check here if user owns the imageId, though Python might do it too
        log.info("Proxying request to list prediction artifacts for user {}, image {}, model_exp {}, path: '{}'",
                currentUsername, imageId, modelExperimentRunId, path);
        List<Map<String, Object>> artifacts = pythonApiService.listPythonPredictionArtifacts(
                currentUsername, String.valueOf(imageId), modelExperimentRunId, path
        );
        return ResponseEntity.ok(artifacts);
    }

    @GetMapping("/{imageId}/model/{modelExperimentRunId}/artifacts/content/**")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Resource> getPredictionArtifactContent(
            @PathVariable Long imageId,
            @PathVariable String modelExperimentRunId,
            HttpServletRequest request,
            Authentication authentication) {
        String currentUsername = authentication.getName();
        // Optional: Check ownership

        String fullRequestPath = urlPathHelper.getPathWithinApplication(request); // More robust
        String basePathToTrim = String.format("/api/predictions/%d/model/%s/artifacts/content/", imageId, modelExperimentRunId);
        String artifactRelativePath = "";

        if (fullRequestPath.startsWith(basePathToTrim)) {
            artifactRelativePath = fullRequestPath.substring(basePathToTrim.length());
        } else {
            log.error("Could not extract artifact relative path from request URI: {}", fullRequestPath);
            return ResponseEntity.badRequest().build();
        }

        if (artifactRelativePath.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        log.info("Proxying request for prediction artifact content: user {}, image {}, model_exp {}, relPath: {}",
                currentUsername, imageId, modelExperimentRunId, artifactRelativePath);
        try {
            byte[] contentBytes = pythonApiService.getPythonPredictionArtifactContent(
                    currentUsername, String.valueOf(imageId), modelExperimentRunId, artifactRelativePath
            );

            HttpHeaders headers = new HttpHeaders();
            String filename = Paths.get(artifactRelativePath).getFileName().toString();
            String mediaType = "application/octet-stream";

            if (filename.toLowerCase().endsWith(".json")) mediaType = MediaType.APPLICATION_JSON_VALUE;
            else if (filename.toLowerCase().endsWith(".png")) mediaType = MediaType.IMAGE_PNG_VALUE;
            else if (filename.toLowerCase().endsWith(".jpg") || filename.toLowerCase().endsWith(".jpeg")) mediaType = MediaType.IMAGE_JPEG_VALUE;

            headers.setContentType(MediaType.parseMediaType(mediaType));
            headers.setContentDispositionFormData("inline", filename);

            ByteArrayResource resource = new ByteArrayResource(contentBytes);
            return ResponseEntity.ok().headers(headers).contentLength(contentBytes.length).body(resource);
        } catch (ResourceNotFoundException e) {
            log.error("Artifact not found: user {}, image {}, model_exp {}, relPath: {}. Error: {}",
                    currentUsername, imageId, modelExperimentRunId, artifactRelativePath, e.getMessage());
            return ResponseEntity.notFound().build();
        }
        catch (Exception e) {
            log.error("Error retrieving prediction artifact content: user {}, image {}, model_exp {}, relPath: {}. Error: {}",
                    currentUsername, imageId, modelExperimentRunId, artifactRelativePath, e.getMessage());
        }
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build(); // Fallback
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
