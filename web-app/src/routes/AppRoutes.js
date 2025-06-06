import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import HomePage from '../pages/HomePage';
import LoginPage from '../pages/LoginPage';
import RegisterPage from '../pages/RegisterPage';
import SettingsPage from '../pages/SettingsPage';
import UploadedImagesPage from '../pages/UploadedImagesPage';
import ViewImagePredictionsPage from '../pages/ViewImagePredictionsPage';
import ExperimentsPage from '../pages/ExperimentsPage';
import CreateExperimentPage from '../pages/CreateExperimentPage';
import ViewExperimentPage from '../pages/ViewExperimentPage';
import PrivateRoute from './PrivateRoute';
import useAuth from '../hooks/useAuth';

const AppRoutes = () => {
    const { user } = useAuth();

    return (
        <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />

            <Route element={<Layout />}> {/* Routes within Layout will have Sidenav and TopBar */}
                <Route path="/" element={<PrivateRoute><HomePage /></PrivateRoute>} />
                <Route path="/settings" element={<PrivateRoute><SettingsPage /></PrivateRoute>} />
                <Route path="/images" element={<PrivateRoute><UploadedImagesPage /></PrivateRoute>} />
                <Route path="/images/:imageId/predictions" element={<PrivateRoute><ViewImagePredictionsPage /></PrivateRoute>} />

                {/* Meteorologist Routes */}
                <Route
                    path="/experiments"
                    element={
                        <PrivateRoute requiredRole="METEOROLOGIST">
                            <ExperimentsPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/experiments/create"
                    element={
                        <PrivateRoute requiredRole="METEOROLOGIST">
                            <CreateExperimentPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/experiments/:experimentRunId" // Match Java's experimentRunId
                    element={
                        <PrivateRoute requiredRole="METEOROLOGIST">
                            <ViewExperimentPage />
                        </PrivateRoute>
                    }
                />
                {/* Fallback for logged-in users if no other route matches */}
                <Route path="*" element={<PrivateRoute><Navigate to="/" replace /></PrivateRoute>} />
            </Route>
        </Routes>
    );
};

export default AppRoutes;