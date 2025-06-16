import axios from 'axios';
import {API_BASE_URL} from '../config';

const API_URL = `${API_BASE_URL}/auth`;

const setAuthHeader = (token) => {
    if (token) {
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
        delete axios.defaults.headers.common['Authorization'];
    }
};

const initialToken = localStorage.getItem('authToken');
if (initialToken) {
    setAuthHeader(initialToken);
}


const login = async (username, password) => {
    const response = await axios.post(`${API_URL}/login`, { username, password });
    if (response.data.token) {
        setAuthHeader(response.data.token);
    }
    return response.data;
};

const register = async (userData) => {
    const response = await axios.post(`${API_URL}/register`, userData);
    return response.data;
};

const logout = () => {
    setAuthHeader(null);
};

const getCurrentUser = async (token) => {
    const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
    const response = await axios.get(`${API_BASE_URL}/users/me`, config);
    return response.data;
};


const authService = {
    login,
    register,
    logout,
    getCurrentUser,
    setAuthHeader
};

export default authService;