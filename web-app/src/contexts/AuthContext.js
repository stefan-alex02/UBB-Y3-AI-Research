import React, { createContext, useState, useEffect, useCallback } from 'react';
import authService from '../services/authService'; // We'll create this

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null); // Stores { id, username, role, token }
    const [loading, setLoading] = useState(true);

    const initializeAuth = useCallback(async () => {
        const token = localStorage.getItem('authToken');
        if (token) {
            try {
                // Optionally, validate token with backend here or just decode
                const userData = await authService.getCurrentUser(token); // Assumes service handles token attachment
                setUser({ ...userData, token });
            } catch (error) {
                console.error("Token validation failed or user fetch failed", error);
                localStorage.removeItem('authToken');
                setUser(null);
            }
        }
        setLoading(false);
    }, []);

    useEffect(() => {
        initializeAuth();
    }, [initializeAuth]);

    const login = async (username, password) => {
        const data = await authService.login(username, password);
        localStorage.setItem('authToken', data.token);
        setUser({ id: data.id, username: data.username, role: data.role, token: data.token });
        return data; // Return full response for potential redirects
    };

    const register = async (userData) => {
        // Register service might not return a token immediately, or might log in
        await authService.register(userData);
        // Optionally log in the user after registration or prompt them to log in
    };

    const logout = () => {
        localStorage.removeItem('authToken');
        setUser(null);
        authService.logout(); // Clear axios auth header if set
    };

    const value = { user, loading, login, register, logout };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export default AuthContext;