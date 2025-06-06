import axios from 'axios';
import { API_BASE_URL } from '../config';

const API_URL = `${API_BASE_URL}/users`;

// getCurrentUser is handled by authService as it's closely tied to auth state

const updateUserSettings = async (userId, settingsData) => {
    // settingsData: { name, newPassword }
    // Auth token should be automatically included by axios interceptor or authService.setAuthHeader
    const response = await axios.put(`${API_URL}/${userId}/settings`, settingsData);
    return response.data; // Expects UserDTO
};

const userService = {
    updateUserSettings,
};

export default userService;