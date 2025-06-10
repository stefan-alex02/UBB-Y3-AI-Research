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

const listPredictionArtifacts = async (username, imageId, experimentIdOfModel, subPath = '') => {
    const params = subPath ? { path: subPath } : {};
    // Calls Java proxy endpoint
    const response = await axios.get(`${API_URL}/${imageId}/model/${experimentIdOfModel}/artifacts/list`, { params });
    return response.data;
};

const getPredictionArtifactContent = async (username, imageId, experimentIdOfModel, artifactRelativePath) => {
    // Calls Java proxy endpoint
    const artifactUrl = `${API_URL}/${imageId}/model/${experimentIdOfModel}/artifacts/content/${artifactRelativePath}`;
    let responseType = 'blob';
    const ext = artifactRelativePath.split('.').pop()?.toLowerCase();
    if (['json', 'log', 'txt', 'csv'].includes(ext)) {
        responseType = 'text';
    }
    const response = await axios.get(artifactUrl, { responseType: responseType });
    return response.data;
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