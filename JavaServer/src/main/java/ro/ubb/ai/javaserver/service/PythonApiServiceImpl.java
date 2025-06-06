package ro.ubb.ai.javaserver.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import ro.ubb.ai.javaserver.dto.experiment.PythonExperimentRunResponseDTO;
import ro.ubb.ai.javaserver.dto.experiment.PythonRunExperimentRequestDTO;
import ro.ubb.ai.javaserver.dto.prediction.PythonPredictionRequestDTO;
import ro.ubb.ai.javaserver.dto.prediction.PythonPredictionRunResponseDTO;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;

import java.util.List;
import java.util.Map;

@Service
@Slf4j
public class PythonApiServiceImpl implements PythonApiService {

    private final RestTemplate restTemplate;
    private final String pythonApiBaseUrl;
    private final String internalApiKey; // Key for Java -> Python internal calls (if any)
    private final ObjectMapper objectMapper;


    public PythonApiServiceImpl(RestTemplate restTemplate,
                                @Value("${python.api.base-url}") String pythonApiBaseUrl,
                                @Value("${python.api.internal-key}") String internalApiKey,
                                ObjectMapper objectMapper) {
        this.restTemplate = restTemplate;
        this.pythonApiBaseUrl = pythonApiBaseUrl;
        this.internalApiKey = internalApiKey; // Not used in current example calls to Python
        this.objectMapper = objectMapper;
    }

    @Override
    public PythonExperimentRunResponseDTO startPythonExperiment(PythonRunExperimentRequestDTO requestDTO) {
        String url = pythonApiBaseUrl + "/experiments/run";
        log.info("Sending request to Python to start experiment: {} with ID {}", requestDTO.getDatasetName(), requestDTO.getExperimentRunId());
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            // If Python's /experiments/run endpoint is protected by X-Internal-API-Key:
            // headers.set("X-Internal-API-Key", internalApiKey);

            HttpEntity<PythonRunExperimentRequestDTO> entity = new HttpEntity<>(requestDTO, headers);
            ResponseEntity<PythonExperimentRunResponseDTO> response = restTemplate.postForEntity(url, entity, PythonExperimentRunResponseDTO.class);
            log.info("Received response from Python for experiment start: {}", response.getBody());
            return response.getBody();
        } catch (HttpClientErrorException e) {
            log.error("Error starting Python experiment {}: {} - {}", requestDTO.getExperimentRunId(), e.getStatusCode(), e.getResponseBodyAsString());
            // Convert Python's error to a more generic message or rethrow a custom exception
            throw new RuntimeException("Failed to start experiment in Python: " + e.getResponseBodyAsString(), e);
        } catch (Exception e) {
            log.error("Unexpected error starting Python experiment {}: {}", requestDTO.getExperimentRunId(), e.getMessage(), e);
            throw new RuntimeException("Unexpected error communicating with Python service.", e);
        }
    }

    @Override
    public List<Map<String, Object>> listPythonExperimentArtifacts(String datasetName, String modelType, String experimentRunId, String subPath) {
        // subPath is relative to the experiment_run_id folder. Example: "single_train_0/plots" or "" for root
        String url = String.format("%s/experiments/%s/%s/%s/artifacts", pythonApiBaseUrl, datasetName, modelType, experimentRunId);
        if (subPath != null && !subPath.isEmpty()) {
            url += "?prefix=" + subPath;
        }
        log.info("Listing artifacts from Python for experiment: {} path: {}", experimentRunId, subPath);
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return objectMapper.readValue(response.getBody(), new TypeReference<List<Map<String, Object>>>() {});
            } else {
                log.error("Failed to list artifacts from Python. Status: {}, Body: {}", response.getStatusCode(), response.getBody());
                throw new RuntimeException("Failed to list artifacts from Python: " + response.getBody());
            }
        } catch (Exception e) {
            log.error("Error listing artifacts from Python for experiment {}: {}", experimentRunId, e.getMessage(), e);
            throw new RuntimeException("Error listing artifacts from Python.", e);
        }
    }

    @Override
    public PythonPredictionRunResponseDTO runPredictionInPython(PythonPredictionRequestDTO requestDTO) {
        String url = pythonApiBaseUrl + "/predictions/run";
        log.info("Sending prediction request to Python for user: {}", requestDTO.getUsername());
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<PythonPredictionRequestDTO> entity = new HttpEntity<>(requestDTO, headers);
            ResponseEntity<PythonPredictionRunResponseDTO> response = restTemplate.postForEntity(url, entity, PythonPredictionRunResponseDTO.class);
            return response.getBody();
        } catch (HttpClientErrorException e) {
            log.error("Error running prediction in Python for user {}: {} - {}", requestDTO.getUsername(), e.getStatusCode(), e.getResponseBodyAsString());
            throw new RuntimeException("Prediction failed in Python: " + e.getResponseBodyAsString(), e);
        } catch (Exception e) {
            log.error("Unexpected error during Python prediction for user {}: {}", requestDTO.getUsername(), e.getMessage(), e);
            throw new RuntimeException("Unexpected error communicating with Python service for prediction.", e);
        }
    }

    @Override
    public void uploadImageToPython(String username, String imageId, String imageFormat, MultipartFile file) {
        String url = pythonApiBaseUrl + "/images/upload";
        log.info("Uploading image {} (ID: {}) with format {} to Python for user {}",
                file.getOriginalFilename(), imageId, imageFormat, username); // Added imageFormat to log
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("username", username);
            body.add("image_id", imageId);
            body.add("image_format", imageFormat);

            // --- SIMPLIFIED FILE PART ADDITION ---
            // Wrap the MultipartFile's bytes in a ByteArrayResource.
            // Crucially, override getFilename() so RestTemplate can set the
            // Content-Disposition filename parameter for the file part.
            ByteArrayResource fileResource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename(); // Ensures the filename is part of the Content-Disposition
                }
            };
            body.add("file", fileResource); // "file" must match the FastAPI parameter name
            // --- END OF SIMPLIFIED FILE PART ---


            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            log.debug("Sending multipart request to Python. Keys in body: {}", body.keySet()); // Log what keys Java is adding

            ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);

            if (!response.getStatusCode().is2xxSuccessful()) {
                log.error("Failed to upload image to Python (non-2xx): {} - {}", response.getStatusCode(), response.getBody());
                throw new RuntimeException("Failed to upload image to Python: " + response.getBody());
            }
            log.info("Image {} (ID: {}) successfully sent to Python for user {}", file.getOriginalFilename(), imageId, username);

        } catch (HttpClientErrorException e) {
            log.error("HttpClientError uploading image {} to Python: {} - Response: {}", file.getOriginalFilename(), e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new RuntimeException("Failed to upload image to Python (client error): " + e.getResponseBodyAsString(), e);
        } catch (Exception e) {
            log.error("Error uploading image {} to Python: {}", file.getOriginalFilename(), e.getMessage(), e);
            throw new RuntimeException("Failed to upload image to Python.", e);
        }
    }

    @Override
    public byte[] downloadImageFromPython(String username, String imageFilenameWithExt) {
        String url = String.format("%s/images/%s/%s", pythonApiBaseUrl, username, imageFilenameWithExt);
        log.info("Requesting image content from Python: {}", url);
        try {
            // Requesting as byte array directly
            ResponseEntity<byte[]> response = restTemplate.exchange(url, HttpMethod.GET, null, byte[].class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                log.debug("Successfully received image bytes from Python for: {}", imageFilenameWithExt);
                return response.getBody();
            } else {
                log.error("Failed to download image from Python. Status: {}, URL: {}", response.getStatusCode(), url);
                throw new RuntimeException("Failed to download image from Python: " + response.getStatusCode());
            }
        } catch (HttpClientErrorException e) {
            log.error("HttpClientError downloading image from Python {}: {} - {}", url, e.getStatusCode(), e.getResponseBodyAsString());
            if (e.getStatusCode() == HttpStatus.NOT_FOUND) {
                throw new ResourceNotFoundException("Image file not found on Python server: " + imageFilenameWithExt);
            }
            throw new RuntimeException("Client error downloading image from Python: " + e.getResponseBodyAsString(), e);
        } catch (Exception e) {
            log.error("Error downloading image from Python {}: {}", url, e.getMessage(), e);
            throw new RuntimeException("Error communicating with Python service for image download.", e);
        }
    }

    // Similar methods for listPythonPredictionArtifacts, getPythonArtifact (returning byte[] or Resource), deletePythonArtifacts etc.
}