import axios from 'axios';
import { API_BASE_URL } from '../config';

const API_URL = `${API_BASE_URL}/auth`;

// Helper to set Authorization header
const setAuthHeader = (token) => {
    if (token) {
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
        delete axios.defaults.headers.common['Authorization'];
    }
};

// Initialize auth header if token exists in localStorage
const initialToken = localStorage.getItem('authToken');
if (initialToken) {
    setAuthHeader(initialToken);
}


const login = async (username, password) => {
    const response = await axios.post(`${API_URL}/login`, { username, password });
    if (response.data.token) {
        setAuthHeader(response.data.token);
    }
    return response.data; // Expected: { token, id, username, role }
};

const register = async (userData) => { // { username, password, name, role, meteorologistPasscode }
    const response = await axios.post(`${API_URL}/register`, userData);
    return response.data; // Expected: success message or user object
};

const logout = () => {
    setAuthHeader(null); // Clear auth header
};

const getCurrentUser = async (token) => {
    // This service assumes token is already set in axios header by AuthContext
    // Or, if called before context is fully set, pass token explicitly
    const config = token ? { headers: { Authorization: `Bearer ${token}` } } : {};
    const response = await axios.get(`${API_BASE_URL}/users/me`, config);
    return response.data; // Expected: { id, username, name, role, createdAt }
};


const authService = {
    login,
    register,
    logout,
    getCurrentUser,
    setAuthHeader // Expose if needed elsewhere, though AuthContext manages it
};

export default authService;