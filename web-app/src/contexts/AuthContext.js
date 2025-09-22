import React, {createContext, useCallback, useEffect, useState} from 'react';
import authService from '../services/authService';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    const initializeAuth = useCallback(async () => {
        const token = localStorage.getItem('authToken');
        if (token) {
            try {
                const userData = await authService.getCurrentUser(token);
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
        setUser({ id: data.id, name: data.name, username: data.username, role: data.role, token: data.token });
        return data;
    };

    const register = async (userData) => {
        await authService.register(userData);
    };

    const logout = () => {
        localStorage.removeItem('authToken');
        setUser(null);
        authService.logout();
    };

    const value = { user, setUser, loading, login, register, logout };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export default AuthContext;