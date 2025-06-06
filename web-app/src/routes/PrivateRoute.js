import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import useAuth from '../hooks/useAuth';
import LoadingSpinner from '../components/LoadingSpinner'; // You'll create this

const PrivateRoute = ({ children, requiredRole }) => {
    const { user, loading } = useAuth();
    const location = useLocation();

    if (loading) {
        return <LoadingSpinner />; // Or some other loading indicator
    }

    if (!user) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    if (requiredRole && user.role !== requiredRole) {
        // Optionally, redirect to an "Unauthorized" page or home
        return <Navigate to="/" state={{ from: location, error: "unauthorized" }} replace />;
    }

    return children;
};

export default PrivateRoute;