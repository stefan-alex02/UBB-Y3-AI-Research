import React from 'react';
import { Container, Typography, Paper, Box } from '@mui/material';
import useAuth from '../hooks/useAuth';

const HomePage = () => {
    const { user } = useAuth();

    return (
        <Container maxWidth="md">
            <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
                <Typography variant="h4" component="h1" gutterBottom>
                    Welcome to Cloud Classifier, {user?.name || user?.username}!
                </Typography>
                <Typography variant="body1" sx={{ mb: 2 }}>
                    This is your central dashboard. Use the navigation menu on the left to access different features of the application.
                </Typography>
                {user?.role === 'METEOROLOGIST' && (
                    <Box>
                        <Typography variant="h6" gutterBottom>Quick Links for Meteorologists:</Typography>
                        <ul>
                            <li><Typography component="a" href="/experiments" color="primary">View Experiments</Typography></li>
                            <li><Typography component="a" href="/experiments/create" color="primary">Create New Experiment</Typography></li>
                        </ul>
                    </Box>
                )}
                <Box mt={2}>
                    <Typography variant="h6" gutterBottom>Common Actions:</Typography>
                    <ul>
                        <li><Typography component="a" href="/images" color="primary">View Your Uploaded Images</Typography></li>
                        <li><Typography component="a" href="/settings" color="primary">Account Settings</Typography></li>
                    </ul>
                </Box>
            </Paper>
        </Container>
    );
};

export default HomePage;