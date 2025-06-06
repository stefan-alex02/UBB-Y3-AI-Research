import React, { useState, useEffect } from 'react';
import { Container, Typography, Paper, Box, TextField, Button, Alert, CircularProgress, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import useAuth from '../hooks/useAuth';
import userService from '../services/userService';
import { useThemeMode } from '../contexts/ThemeContext';

const SettingsSchema = Yup.object().shape({
    name: Yup.string().max(100, 'Too Long!'),
    newPassword: Yup.string().min(6, 'Password must be at least 6 characters').optional(),
    confirmNewPassword: Yup.string()
        .when('newPassword', (newPassword, schema) => {
            // Access the first element of newPassword array if it's an array
            const passwordValue = Array.isArray(newPassword) ? newPassword[0] : newPassword;
            return passwordValue ? schema.required('Confirm new password is required').oneOf([Yup.ref('newPassword')], 'Passwords must match') : schema;
        }),
});


const SettingsPage = () => {
    const { user, setUser } = useAuth(); // Assuming setUser in AuthContext can update user details
    const { mode, setThemeMode } = useThemeMode();
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [loading, setLoading] = useState(false);
    const [currentThemePreference, setCurrentThemePreference] = useState(localStorage.getItem('themeModePreference') || 'system');


    const handleThemeChange = (event) => {
        const newPreference = event.target.value;
        setCurrentThemePreference(newPreference);
        localStorage.setItem('themeModePreference', newPreference);
        setThemeMode(newPreference); // This calls the context function
    };


    if (!user) return <CircularProgress />;

    return (
        <Container maxWidth="sm">
            <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
                <Typography variant="h4" component="h1" gutterBottom>
                    Account Settings
                </Typography>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

                <Formik
                    initialValues={{
                        name: user.name || '',
                        newPassword: '',
                        confirmNewPassword: '',
                    }}
                    validationSchema={SettingsSchema}
                    enableReinitialize // Important to reinitialize form when user data changes
                    onSubmit={async (values, { setSubmitting, resetForm }) => {
                        setError('');
                        setSuccess('');
                        setLoading(true);
                        const updateData = {};
                        if (values.name !== user.name) updateData.name = values.name;
                        if (values.newPassword) updateData.newPassword = values.newPassword;

                        if (Object.keys(updateData).length === 0) {
                            setSuccess("No changes to save.");
                            setLoading(false);
                            setSubmitting(false);
                            return;
                        }

                        try {
                            const updatedUserDTO = await userService.updateUserSettings(user.id, updateData);
                            // Update user in AuthContext if your context supports it
                            if (setUser) {
                                setUser(prevUser => ({...prevUser, name: updatedUserDTO.name}));
                            }
                            setSuccess('Settings updated successfully!');
                            resetForm({ values: { name: updatedUserDTO.name || '', newPassword: '', confirmNewPassword: '' } });
                        } catch (err) {
                            setError(err.response?.data?.message || err.message || 'Failed to update settings.');
                        } finally {
                            setLoading(false);
                            setSubmitting(false);
                        }
                    }}
                >
                    {({ errors, touched, isSubmitting }) => (
                        <Form noValidate>
                            <Field
                                as={TextField}
                                name="name"
                                label="Full Name"
                                fullWidth
                                margin="normal"
                                error={touched.name && Boolean(errors.name)}
                                helperText={touched.name && errors.name}
                                disabled={loading}
                            />
                            <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>Change Password (optional)</Typography>
                            <Field
                                as={TextField}
                                name="newPassword"
                                label="New Password"
                                type="password"
                                fullWidth
                                margin="normal"
                                error={touched.newPassword && Boolean(errors.newPassword)}
                                helperText={touched.newPassword && errors.newPassword}
                                disabled={loading}
                            />
                            <Field
                                as={TextField}
                                name="confirmNewPassword"
                                label="Confirm New Password"
                                type="password"
                                fullWidth
                                margin="normal"
                                error={touched.confirmNewPassword && Boolean(errors.confirmNewPassword)}
                                helperText={touched.confirmNewPassword && errors.confirmNewPassword}
                                disabled={loading}
                            />
                            <Button
                                type="submit"
                                variant="contained"
                                sx={{ mt: 2 }}
                                disabled={isSubmitting || loading}
                            >
                                {loading ? <CircularProgress size={24} /> : 'Save Profile Changes'}
                            </Button>
                        </Form>
                    )}
                </Formik>

                <Box sx={{ mt: 4 }}>
                    <Typography variant="h6" gutterBottom>Theme Preference</Typography>
                    <FormControl fullWidth margin="normal">
                        <InputLabel id="theme-select-label">Theme</InputLabel>
                        <Select
                            labelId="theme-select-label"
                            id="theme-select"
                            value={currentThemePreference}
                            label="Theme"
                            onChange={handleThemeChange}
                        >
                            <MenuItem value="light">Light</MenuItem>
                            <MenuItem value="dark">Dark</MenuItem>
                            <MenuItem value="system">System Default</MenuItem>
                        </Select>
                    </FormControl>
                </Box>
            </Paper>
        </Container>
    );
};

export default SettingsPage;