import axios from 'axios';
import {API_BASE_URL} from '../config';

const API_URL = `${API_BASE_URL}/experiments`;

const createExperiment = async (experimentData) => {
    console.log('Creating experiment with data:', experimentData);
    const response = await axios.post(API_URL, experimentData);
    return response.data;
};

const getExperiments = async (filters = {}, pageable = { page: 0, size: 10, sortBy: 'startTime', sortDir: 'DESC' }) => {
    const params = {
        ...filters,
        page: pageable.page,
        size: pageable.size,
        sort: `${pageable.sortBy},${pageable.sortDir}`
    };
    Object.keys(params).forEach(key => {
        if (params[key] === null || params[key] === '') {
            delete params[key];
        }
    });
    const response = await axios.get(API_URL, { params });
    return response.data;
};

const getExperimentDetails = async (experimentRunId) => {
    const response = await axios.get(`${API_URL}/${experimentRunId}`);
    return response.data;
};

const deleteExperiment = async (experimentRunId) => {
    await axios.delete(`${API_URL}/${experimentRunId}`);
};

const listExperimentArtifacts = async (experimentRunId, subPath = '') => {
    const params = subPath ? { path: subPath } : {};
    const response = await axios.get(`${API_URL}/${experimentRunId}/artifacts/list`, { params });
    return response.data;
};

const getExperimentArtifactContent = async (experimentRunId, artifactRelativePath) => {
    const artifactUrl = `${API_URL}/${experimentRunId}/artifacts/content/${artifactRelativePath}`;

    let responseType = 'blob';
    const ext = artifactRelativePath.split('.').pop()?.toLowerCase();
    if (['json', 'log', 'txt', 'csv'].includes(ext)) {
        responseType = 'text';
    }

    const response = await axios.get(artifactUrl, { responseType: responseType });
    return response.data;
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