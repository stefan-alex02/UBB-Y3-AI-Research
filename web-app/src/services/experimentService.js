import axios from 'axios';
import { API_BASE_URL } from '../config';

const API_URL = `${API_BASE_URL}/experiments`;

const createExperiment = async (experimentData) => {
    // experimentData: { name, modelType, datasetName, methodsSequence, ... }
    const response = await axios.post(API_URL, experimentData);
    return response.data; // Expects ExperimentDTO
};

const getExperiments = async (filters, pageable) => {
    // filters: { modelType, datasetName, status, startedAfter, finishedBefore }
    // pageable: { page, size, sortBy, sortDir }
    const params = { ...filters, ...pageable };
    const response = await axios.get(API_URL, { params });
    return response.data; // Expects Page<ExperimentDTO>
};

const getExperimentDetails = async (experimentRunId) => {
    const response = await axios.get(`${API_URL}/${experimentRunId}`);
    return response.data; // Expects ExperimentDTO
};

// For fetching artifacts (JSON, text logs)
const getExperimentArtifactContent = async (datasetName, modelType, experimentRunId, artifactPath) => {
    // The actual URL to Python's artifact endpoint, possibly proxied by Java or direct
    // This might be better handled by a generic artifact service if paths are consistent
    const artifactUrl = `${API_BASE_URL}/python-proxy-artifacts/experiments/${datasetName}/${modelType}/${experimentRunId}/${artifactPath}`;
    const response = await axios.get(artifactUrl, { responseType: 'blob' }); // Get as blob for images, or text for json/log

    const contentType = response.headers['content-type'];
    if (contentType && (contentType.includes('application/json') || contentType.includes('text/plain'))) {
        return await response.data.text(); // For JSON or text
    }
    return response.data; // For images, this will be a Blob
};

// For getting artifact list
const listExperimentArtifacts = async (datasetName, modelType, experimentRunId, subPath = '') => {
    const params = subPath ? { prefix: subPath } : {};
    const response = await axios.get(`${API_URL}/python-proxy-artifacts/experiments/${datasetName}/${modelType}/${experimentRunId}/list`, { params });
    return response.data; // Expects List<ArtifactNode>
}


const experimentService = {
    createExperiment,
    getExperiments,
    getExperimentDetails,
    getExperimentArtifactContent,
    listExperimentArtifacts,
    // deleteExperiment
};

export default experimentService;