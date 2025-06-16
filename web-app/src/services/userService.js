import axios from 'axios';
import {API_BASE_URL} from '../config';

const API_URL = `${API_BASE_URL}/users`;


const updateUserSettings = async (userId, settingsData) => {
    const response = await axios.put(`${API_URL}/${userId}/settings`, settingsData);
    return response.data;
};

const userService = {
    updateUserSettings,
};

export default userService;