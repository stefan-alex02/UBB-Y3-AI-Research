import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import { TextField, Button, Container, Typography, Box, Alert, CircularProgress } from '@mui/material';
import useAuth from '../hooks/useAuth';

const LoginSchema = Yup.object().shape({
    username: Yup.string().required('Username is required'),
    password: Yup.string().required('Password is required'),
});

const LoginPage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const { login } = useAuth();
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const from = location.state?.from?.pathname || '/';

    return (
        <Container component="main" maxWidth="xs">
            <Box
                sx={{
                    marginTop: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}
            >
                <Typography component="h1" variant="h5">
                    Sign in
                </Typography>
                {error && <Alert severity="error" sx={{ width: '100%', mt: 2 }}>{error}</Alert>}
                <Formik
                    initialValues={{ username: '', password: '' }}
                    validationSchema={LoginSchema}
                    onSubmit={async (values, { setSubmitting }) => {
                        setError('');
                        setLoading(true);
                        try {
                            await login(values.username, values.password);
                            navigate(from, { replace: true });
                        } catch (err) {
                            setError(err.response?.data?.message || err.message || 'Login failed. Please try again.');
                            setLoading(false);
                        }
                        setSubmitting(false);
                    }}
                >
                    {({ errors, touched, isSubmitting }) => (
                        <Form noValidate style={{ width: '100%', marginTop: '1rem' }}>
                            <Field
                                as={TextField}
                                variant="outlined"
                                margin="normal"
                                required
                                fullWidth
                                id="username"
                                label="Username"
                                name="username"
                                autoComplete="username"
                                autoFocus
                                error={touched.username && Boolean(errors.username)}
                                helperText={touched.username && errors.username}
                                disabled={loading}
                            />
                            <Field
                                as={TextField}
                                variant="outlined"
                                margin="normal"
                                required
                                fullWidth
                                name="password"
                                label="Password"
                                type="password"
                                id="password"
                                autoComplete="current-password"
                                error={touched.password && Boolean(errors.password)}
                                helperText={touched.password && errors.password}
                                disabled={loading}
                            />
                            <Button
                                type="submit"
                                fullWidth
                                variant="contained"
                                sx={{ mt: 3, mb: 2 }}
                                disabled={isSubmitting || loading}
                            >
                                {loading ? <CircularProgress size={24} color="inherit" /> : "Sign In"}
                            </Button>
                            <Button fullWidth onClick={() => navigate('/register')}>
                                Don't have an account? Sign Up
                            </Button>
                        </Form>
                    )}
                </Formik>
            </Box>
        </Container>
    );
};

export default LoginPage;