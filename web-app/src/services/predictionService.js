import axios from 'axios';
import { API_BASE_URL } from '../config';

const API_URL = `${API_BASE_URL}/predictions`;

const createPrediction = async (predictionRequestData) => {
    console.log('Creating prediction with data:', predictionRequestData);
    const response = await axios.post(API_URL, predictionRequestData);
    return response.data;
};

const getPredictionsForImage = async (imageId) => {
    const response = await axios.get(`${API_URL}/image/${imageId}`);
    return response.data;
};

const getSpecificPrediction = async (predictionId) => {
    const response = await axios.get(`${API_URL}/${predictionId}`);
    return response.data;
};

const deletePrediction = async (predictionId) => {
    await axios.delete(`${API_URL}/${predictionId}`);
};

const listPredictionArtifacts = async (predictionId, subPath = '') => {
    const params = subPath ? { path: subPath } : {};
    const response = await axios.get(`${API_URL}/${predictionId}/artifacts/list`, { params });
    return response.data;
};

const getPredictionArtifactContent = async (predictionId, artifactRelativePath) => {
    const artifactUrl = `${API_URL}/${predictionId}/artifacts/content/${artifactRelativePath}`;
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