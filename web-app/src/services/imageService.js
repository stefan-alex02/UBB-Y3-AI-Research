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

const deleteImage = async (imageId) => {
    await axios.delete(`${API_URL}/${imageId}`);
};

const getImageDataUrl = (username, imageId, format) => {
    // For displaying image from Python server via Java proxy (if you build one)
    // OR directly to Python if CORS is set up and Python has an image serving endpoint
    return `${API_BASE_URL}/python-proxy-images/${username}/${imageId}.${format}`; // Example proxy path
    // Or if Python serves directly: `PYTHON_API_URL/images/${username}/${imageId}.${format}`
}


const imageService = {
    uploadImage,
    getUserImages,
    deleteImage,
    getImageDataUrl,
};

export default imageService;