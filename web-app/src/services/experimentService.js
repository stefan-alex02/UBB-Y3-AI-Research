import axios from 'axios';
import { API_BASE_URL } from '../config';

const API_URL = `${API_BASE_URL}/experiments`;

const createExperiment = async (experimentData) => {
    // experimentData: { name, modelType, datasetName, methodsSequence, ... }
    console.log('Creating experiment with data:', experimentData);
    const response = await axios.post(API_URL, experimentData);
    return response.data; // Expects ExperimentDTO
};

const getExperiments = async (filters = {}, pageable = { page: 0, size: 10, sortBy: 'startTime', sortDir: 'DESC' }) => {
    const params = {
        ...filters, // modelType, datasetName, status, startedAfter, finishedBefore
        page: pageable.page,
        size: pageable.size,
        sort: `${pageable.sortBy},${pageable.sortDir}` // Spring Pageable expects sort as "property,direction"
    };
    // Remove null or empty string filter values before sending
    Object.keys(params).forEach(key => {
        if (params[key] === null || params[key] === '') {
            delete params[key];
        }
    });
    const response = await axios.get(API_URL, { params });
    return response.data; // Expects Spring Page<ExperimentDTO>
};

const getExperimentDetails = async (experimentRunId) => {
    const response = await axios.get(`${API_URL}/${experimentRunId}`);
    return response.data; // Expects ExperimentDTO
};

const deleteExperiment = async (experimentRunId) => {
    await axios.delete(`${API_URL}/${experimentRunId}`);
};

// This now calls the Java proxy for listing artifacts
const listExperimentArtifacts = async (experimentRunId, subPath = '') => {
    const params = subPath ? { path: subPath } : {}; // Java endpoint expects 'path' query param
    const response = await axios.get(`${API_URL}/${experimentRunId}/artifacts/list`, { params });
    return response.data; // Expects List<ArtifactNode> (Map<String,Object> from Java will be parsed by axios)
};

// This now calls the Java proxy for artifact content
const getExperimentArtifactContent = async (experimentRunId, artifactRelativePath) => {
    // artifactRelativePath is like "method_0/results.json" or "executor_log.log"
    // The Java endpoint /api/experiments/{id}/artifacts/content/** will handle the full path
    const artifactUrl = `${API_URL}/${experimentRunId}/artifacts/content/${artifactRelativePath}`;

    // Determine responseType based on expected content (could be passed in or inferred)
    let responseType = 'blob'; // Default for images/binary
    const ext = artifactRelativePath.split('.').pop()?.toLowerCase();
    if (['json', 'log', 'txt', 'csv'].includes(ext)) {
        responseType = 'text';
    }

    const response = await axios.get(artifactUrl, { responseType: responseType });
    return response.data; // Blob for images, string for text/json/csv
};


const experimentService = {
    createExperiment,
    getExperiments,
    getExperimentDetails,
    getExperimentArtifactContent,
    listExperimentArtifacts,
    deleteExperiment,
};

export default experimentService;