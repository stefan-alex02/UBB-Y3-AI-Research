import React, {useState} from 'react';
import {useNavigate} from 'react-router-dom';
import {Field, Form, Formik} from 'formik';
import * as Yup from 'yup';
import {
    Alert,
    Box,
    Button,
    Checkbox,
    CircularProgress,
    Container,
    FormControlLabel,
    FormGroup,
    TextField,
    Typography
} from '@mui/material';
import authService from '../services/authService';

const RegisterSchema = Yup.object().shape({
    username: Yup.string().min(2, 'Too Short!').max(50, 'Too Long!').required('Username is required'),
    name: Yup.string().max(100, 'Too Long!'),
    password: Yup.string().min(2, 'Password must be at least 6 characters').required('Password is required'), // TODO change to 6 characters
    confirmPassword: Yup.string()
        .oneOf([Yup.ref('password'), null], 'Passwords must match')
        .required('Confirm Password is required'),
    isMeteorologist: Yup.boolean(),
    meteorologistPasscode: Yup.string().when('isMeteorologist', {
        is: true,
        then: (schema) => schema.required('Passcode is required for meteorologist registration'),
        otherwise: (schema) => schema.optional(),
    }),
});

const RegisterPage = () => {
    const navigate = useNavigate();
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [loading, setLoading] = useState(false);

    return (
        <Container component="main" maxWidth="xs">
            <Box sx={{ marginTop: 8, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <Typography component="h1" variant="h5">Sign Up</Typography>
                {error && <Alert severity="error" sx={{ width: '100%', mt: 2 }}>{error}</Alert>}
                {success && <Alert severity="success" sx={{ width: '100%', mt: 2 }}>{success}</Alert>}
                <Formik
                    initialValues={{
                        username: '',
                        name: '',
                        password: '',
                        confirmPassword: '',
                        isMeteorologist: false,
                        meteorologistPasscode: '',
                    }}
                    validationSchema={RegisterSchema}
                    onSubmit={async (values, { setSubmitting, resetForm }) => {
                        setError('');
                        setSuccess('');
                        setLoading(true);
                        const registrationData = {
                            username: values.username,
                            password: values.password,
                            name: values.name,
                            role: values.isMeteorologist ? 'METEOROLOGIST' : 'NORMAL',
                            ...(values.isMeteorologist && { meteorologist_passcode: values.meteorologistPasscode }),
                        };
                        try {
                            await authService.register(registrationData);
                            setSuccess('Registration successful! Please log in.');
                            resetForm();
                        } catch (err) {
                            setError(err.response?.data?.message || err.message || 'Registration failed.');
                        } finally {
                            setLoading(false);
                            setSubmitting(false);
                        }
                    }}
                >
                    {({ errors, touched, values, handleChange, isSubmitting }) => (
                        <Form noValidate style={{ width: '100%', marginTop: '1rem' }}>
                            <Field as={TextField} name="username" label="Username" fullWidth margin="normal" required error={touched.username && Boolean(errors.username)} helperText={touched.username && errors.username} disabled={loading}/>
                            <Field as={TextField} name="name" label="Full Name (Optional)" fullWidth margin="normal" error={touched.name && Boolean(errors.name)} helperText={touched.name && errors.name} disabled={loading}/>
                            <Field as={TextField} name="password" label="Password" type="password" fullWidth margin="normal" required error={touched.password && Boolean(errors.password)} helperText={touched.password && errors.password} disabled={loading}/>
                            <Field as={TextField} name="confirmPassword" label="Confirm Password" type="password" fullWidth margin="normal" required error={touched.confirmPassword && Boolean(errors.confirmPassword)} helperText={touched.confirmPassword && errors.confirmPassword} disabled={loading}/>
                            <FormGroup>
                                <FormControlLabel
                                    control={<Checkbox name="isMeteorologist" checked={values.isMeteorologist} onChange={handleChange} disabled={loading} />}
                                    label="Register as Meteorologist"
                                />
                            </FormGroup>
                            {values.isMeteorologist && (
                                <Field as={TextField} name="meteorologistPasscode" label="Meteorologist Passcode" type="password" fullWidth margin="normal" required error={touched.meteorologistPasscode && Boolean(errors.meteorologistPasscode)} helperText={touched.meteorologistPasscode && errors.meteorologistPasscode} disabled={loading}/>
                            )}
                            <Button type="submit" fullWidth variant="contained" sx={{ mt: 3, mb: 2 }} disabled={isSubmitting || loading}>
                                {loading ? <CircularProgress size={24} color="inherit" /> : "Sign Up"}
                            </Button>
                            <Button fullWidth onClick={() => navigate('/login')}>
                                Already have an account? Sign In
                            </Button>
                        </Form>
                    )}
                </Formik>
            </Box>
        </Container>
    );
};

export default RegisterPage;