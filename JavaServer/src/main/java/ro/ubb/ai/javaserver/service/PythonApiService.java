package ro.ubb.ai.javaserver.service;

import org.springframework.web.multipart.MultipartFile;
import ro.ubb.ai.javaserver.dto.experiment.PythonExperimentRunResponseDTO;
import ro.ubb.ai.javaserver.dto.experiment.PythonRunExperimentRequestDTO;
import ro.ubb.ai.javaserver.dto.prediction.PythonPredictionRequestDTO;
import ro.ubb.ai.javaserver.dto.prediction.PythonPredictionRunResponseDTO;

import java.util.List;
import java.util.Map;


public interface PythonApiService {
    PythonExperimentRunResponseDTO startPythonExperiment(PythonRunExperimentRequestDTO requestDTO);
    List<Map<String, Object>> listPythonExperimentArtifacts(String datasetName, String modelType, String experimentRunId, String path);
    byte[] getPythonExperimentArtifactContent(String datasetName, String modelType, String experimentRunId, String artifactRelativePath);
    void deletePythonExperimentArtifacts(String datasetName, String modelType, String experimentRunId);

    PythonPredictionRunResponseDTO runPredictionInPython(PythonPredictionRequestDTO requestDTO);
    List<Map<String, Object>> listPythonPredictionArtifacts(String username, String imageId, String predictionId, String path);
    byte[] getPythonPredictionArtifactContent(String username, String imageId, String predictionId, String artifactRelativePath);
    void deletePythonPredictionArtifacts(String username, String imageId, String predictionId);

    void uploadImageToPython(String username, String imageId, String imageFormat, MultipartFile file);
    byte[] downloadImageFromPython(String username, String imageFilenameWithExt);
    void deletePythonImage(String username, String imageIdWithFormat);
}
