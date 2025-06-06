import axios from 'axios';
import { API_BASE_URL } from '../config';

const API_URL = `${API_BASE_URL}/images`;

const uploadImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    // Username will be inferred from JWT on backend
    // Format will be inferred from file.type or filename on backend
    // If backend needs explicit username/format in Form:
    // formData.append('username', 'current_user_from_auth_context'); // Not ideal here
    // formData.append('image_format', file.name.split('.').pop());

    const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data; // Expects ImageDTO
};

const getUserImages = async () => {
    const response = await axios.get(API_URL);
    return response.data; // Expects List<ImageDTO>
};

const getImageByIdForUser = async (imageId) => {
    // The Java endpoint /api/images/{imageId} should verify ownership internally
    const response = await axios.get(`${API_URL}/${imageId}`);
    return response.data; // Expects ImageDTO
};

const getImageContentBlob = async (imageId) => {
    // Assumes auth token is set in axios defaults
    const response = await axios.get(`${API_URL}/${imageId}/content`, {
        responseType: 'blob', // Important: Fetch as a Blob
    });
    return response.data; // This will be a Blob
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