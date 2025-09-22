import axios from 'axios';
import {API_BASE_URL} from '../config';

const API_URL = `${API_BASE_URL}/images`;

const uploadImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

const getUserImages = async () => {
    const response = await axios.get(API_URL);
    return response.data;
};

const getImageByIdForUser = async (imageId) => {
    const response = await axios.get(`${API_URL}/${imageId}`);
    return response.data;
};

const getImageContentBlob = async (imageId) => {
    const response = await axios.get(`${API_URL}/${imageId}/content`, {
        responseType: 'blob',
    });
    return response.data;
};

const deleteImage = async (imageId) => {
    await axios.delete(`${API_URL}/${imageId}`);
};


const imageService = {
    uploadImage,
    getUserImages,
    deleteImage,
    getImageContentBlob,
    getImageByIdForUser,
};

export default imageService;