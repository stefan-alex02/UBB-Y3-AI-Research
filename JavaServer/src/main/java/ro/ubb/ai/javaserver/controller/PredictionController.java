package ro.ubb.ai.javaserver.controller;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.util.UrlPathHelper;
import ro.ubb.ai.javaserver.dto.prediction.PredictionCreateRequest;
import ro.ubb.ai.javaserver.dto.prediction.PredictionDTO;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;
import ro.ubb.ai.javaserver.service.PredictionService;
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
    private final PythonApiService pythonApiService;
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

    @GetMapping("/{predictionId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<PredictionDTO> getSpecificPredictionById(
            @PathVariable Long predictionId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        return ResponseEntity.ok(predictionService.getPrediction(predictionId, currentUsername));
    }

    @GetMapping("/{predictionId}/artifacts/list")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<List<Map<String, Object>>> listPredictionArtifacts(
            @PathVariable Long predictionId,
            @RequestParam(required = false, defaultValue = "") String path,
            Authentication authentication) {
        String currentUsername = authentication.getName();
        PredictionDTO prediction = predictionService.getPrediction(predictionId, currentUsername);

        List<Map<String, Object>> artifacts = pythonApiService.listPythonPredictionArtifacts(
                currentUsername, String.valueOf(prediction.getImageId()), String.valueOf(predictionId), path
        );
        return ResponseEntity.ok(artifacts);
    }

    @GetMapping("/{predictionId}/artifacts/content/**")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Resource> getPredictionArtifactContent(
            @PathVariable Long predictionId,
            HttpServletRequest request,
            Authentication authentication) {
        String currentUsername = authentication.getName();
        PredictionDTO prediction = predictionService.getPrediction(predictionId, currentUsername);

        String fullRequestPath = urlPathHelper.getPathWithinApplication(request);
        String basePathToTrim = String.format("/api/predictions/%d/artifacts/content/", predictionId);
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
        log.info("Proxying request for prediction artifact content: user {}, image {}, id {}, relPath: {}",
                currentUsername, prediction.getImageId(), prediction.getId(), artifactRelativePath);
        try {
            byte[] contentBytes = pythonApiService.getPythonPredictionArtifactContent(
                    currentUsername, String.valueOf(prediction.getImageId()), String.valueOf(predictionId), artifactRelativePath
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
            log.error("Artifact not found: user {}, image {}, id {}, relPath: {}. Error: {}",
                    currentUsername, prediction.getImageId(), prediction.getId(), artifactRelativePath, e.getMessage());
            return ResponseEntity.notFound().build();
        }
        catch (Exception e) {
            log.error("Error retrieving prediction artifact content: user {}, image {}, id {}, relPath: {}. Error: {}",
                    currentUsername, prediction.getImageId(), prediction.getId(), artifactRelativePath, e.getMessage());
        }
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
    }

    @DeleteMapping("/{predictionId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Void> deletePrediction(@PathVariable Long predictionId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        predictionService.deletePrediction(predictionId, currentUsername);
        return ResponseEntity.noContent().build();
    }
}
