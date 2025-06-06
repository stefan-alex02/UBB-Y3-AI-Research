import axios from 'axios';
import { API_BASE_URL } from '../config';

const API_URL = `${API_BASE_URL}/predictions`;

const createPrediction = async (predictionRequestData) => {
    // predictionRequestData: { imageId, modelExperimentRunId, generateLime, ... }
    const response = await axios.post(API_URL, predictionRequestData);
    return response.data; // Expects PredictionDTO
};

const getPredictionsForImage = async (imageId) => {
    const response = await axios.get(`${API_URL}/image/${imageId}`);
    return response.data; // Expects List<PredictionDTO>
};

const getSpecificPrediction = async (imageId, modelExperimentRunId) => {
    const response = await axios.get(`${API_URL}/image/${imageId}/model/${modelExperimentRunId}`);
    return response.data; // Expects PredictionDTO
};

const deletePrediction = async (imageId, modelExperimentRunId) => {
    await axios.delete(`${API_URL}/image/${imageId}/model/${modelExperimentRunId}`);
};

// For fetching prediction artifacts (JSON, plots)
// Similar to experimentService, assumes Java proxy or direct Python call setup
const getPredictionArtifactContent = async (username, imageId, experimentIdOfModel, artifactPath) => {
    const artifactUrl = `${API_BASE_URL}/python-proxy-artifacts/predictions/${username}/${imageId}/${experimentIdOfModel}/${artifactPath}`;
    const response = await axios.get(artifactUrl, { responseType: 'blob' });
    const contentType = response.headers['content-type'];
    if (contentType && (contentType.includes('application/json') || contentType.includes('text/plain'))) {
        return await response.data.text();
    }
    return response.data; // Blob for images
};

const listPredictionArtifacts = async (username, imageId, experimentIdOfModel, subPath = '') => {
    const params = subPath ? { prefix: subPath } : {};
    const response = await axios.get(
        `${API_BASE_URL}/python-proxy-artifacts/predictions/${username}/${imageId}/${experimentIdOfModel}/list`, { params }
    );
    return response.data; // Expects List<ArtifactNode>
};

const predictionService = {
    createPrediction,
    getPredictionsForImage,
    getSpecificPrediction,
    deletePrediction,
    getPredictionArtifactContent,
    listPredictionArtifacts,
};

export default predictionService;