package ro.ubb.ai.javaserver.service;

import ro.ubb.ai.javaserver.dto.experiment.*;
import ro.ubb.ai.javaserver.dto.prediction.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.List;
import java.util.Map;


public interface PythonApiService {
    PythonExperimentRunResponseDTO startPythonExperiment(PythonRunExperimentRequestDTO requestDTO);
    List<Map<String, Object>> listPythonExperimentArtifacts(String datasetName, String modelType, String experimentRunId, String subPath);

    PythonPredictionRunResponseDTO runPredictionInPython(PythonPredictionRequestDTO requestDTO);

    void uploadImageToPython(String username, String imageId, String imageFormat, MultipartFile file);
    byte[] downloadImageFromPython(String username, String imageFilenameWithExt);


    // Methods for fetching artifacts (Python will stream bytes)
    // You'd need similar for predictions:
    // List<Map<String, Object>> listPythonPredictionArtifacts(String username, String imageId, String experimentIdOfModel, String subPath);
    // byte[] getPythonArtifact(String fullArtifactPath); // A generic getter, path constructed by caller

    // Methods for deleting artifacts from Python (MinIO)
    // void deletePythonExperimentArtifacts(String datasetName, String modelType, String experimentRunId);
    // void deletePythonImage(String username, String imageIdWithFormat);
    // void deletePythonPredictionArtifacts(String username, String imageId, String experimentIdOfModel);
}
