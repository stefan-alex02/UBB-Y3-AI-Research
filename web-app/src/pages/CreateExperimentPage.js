import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Formik, Form, Field, FieldArray, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import {
    Container, Typography, Paper, Box, Button, Grid, TextField, Select, MenuItem, InputLabel, FormControl,
    FormHelperText, CircularProgress, Alert, IconButton, Switch, FormControlLabel, Divider
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import experimentService from '../services/experimentService';

// Define available model types and datasets (could be fetched or hardcoded)
const MODEL_TYPES = [
    { value: 'cnn', label: 'Simple CNN' },
    { value: 'pvit', label: 'Pretrained ViT' },
    { value: 'hyvit', label: 'Hybrid ViT (CNN + ViT)' },
    { value: 'swin', label: 'Pretrained Swin Transformer' },
    // Add other model types from your Python ModelType enum
];

const DATASET_NAMES = ['GCD', 'mGCD', 'mGCDf', 'swimcat', 'ccsn', 'Swimcat-extend']; // Match names in Python's DATASET_DICT keys

const AVAILABLE_AUG_STRATEGIES = [
    { value: "DEFAULT_STANDARD", label: "Default Standard"},
    { value: "SKY_ONLY_ROTATION", label: "Sky Only Rotation (GCD/Swimcat)"},
    { value: "CCSN_MODERATE", label: "Ground Aware No Rotation (CCSN)"},
    { value: "NO_AUGMENTATION", label: "No Augmentation"},
    { value: "PAPER_GCD", label: "Paper Replication GCD"},
    { value: "PAPER_CCSN", label: "Paper Replication CCSN"},
];


const PIPELINE_METHODS = [
    { value: 'single_train', label: 'Single Train' },
    { value: 'single_eval', label: 'Single Evaluate' },
    { value: 'non_nested_grid_search', label: 'Non-Nested Grid Search (Tune)' },
    { value: 'nested_grid_search', label: 'Nested Grid Search (Estimate)' },
    { value: 'cv_model_evaluation', label: 'CV Model Evaluation (Fixed Params)' },
    // 'load_model' and 'predict_images' are not typically part of an experiment sequence from UI
];

// Simplified parameter structures for each method type
// In a real app, these could be more dynamic based on model_type
const getMethodDefaultParams = (methodName) => {
    switch (methodName) {
        case 'single_train':
            return { params: { max_epochs: 10, lr: 0.001, batch_size: 32 }, save_model: true, plot_level: 1 };
        case 'single_eval':
            return { plot_level: 2 };
        case 'non_nested_grid_search':
            return { params: { param_grid: { lr: [0.001, 0.0001], batch_size: [16,32] }, cv: 3, scoring: 'accuracy' }, save_best_model: true, plot_level: 1 };
        case 'nested_grid_search':
            return { params: { param_grid: { lr: [0.001] }, outer_cv: 2, inner_cv: 2, scoring: 'accuracy' }, plot_level: 1 };
        case 'cv_model_evaluation':
            return { params: { max_epochs: 10 }, cv: 3, evaluate_on: 'full', plot_level: 1 };
        default:
            return { params: {} };
    }
};

const ExperimentSchema = Yup.object().shape({
    name: Yup.string().required('Experiment name is required').max(255),
    datasetName: Yup.string().required('Dataset is required'),
    modelType: Yup.string().required('Model type is required'),
    methodsSequence: Yup.array()
        .of(
            Yup.object().shape({
                method_name: Yup.string().required('Method is required'),
                // params: Yup.object(), // Can be complex to validate deeply, basic object check
                // For specific param validation, you'd need conditional schemas
                params: Yup.object().test(
                    'json-parsable-params',
                    'Parameters must be a valid JSON string for non-empty param objects.',
                    function (value) {
                        if (value && Object.keys(value).length > 0) { // Only validate if params object is not empty
                            try {
                                JSON.stringify(value); // Test if it can be stringified (basic check for serializability)
                                return true;
                            } catch (e) {
                                return this.createError({ message: `Invalid characters or structure in params: ${e.message}` });
                            }
                        }
                        return true; // Allow empty params object
                    }
                ),
                // Optional fields for some methods
                save_model: Yup.boolean().optional(),
                save_best_model: Yup.boolean().optional(),
                plot_level: Yup.number().min(0).max(2).integer().optional(),
            })
        )
        .min(1, 'At least one method is required in the sequence')
        .required('Method sequence is required'),
    imgSizeH: Yup.number().integer().positive().optional(),
    imgSizeW: Yup.number().integer().positive().optional(),
    offlineAugmentation: Yup.boolean().optional(),
    augmentationStrategyOverride: Yup.string().optional(),
});

const CreateExperimentPage = () => {
    const navigate = useNavigate();
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [submitting, setSubmitting] = useState(false);

    const initialValues = {
        name: '',
        datasetName: DATASET_NAMES[0] || '',
        modelType: MODEL_TYPES[0]?.value || '',
        methodsSequence: [{ method_name: PIPELINE_METHODS[0]?.value || '', ...getMethodDefaultParams(PIPELINE_METHODS[0]?.value) }],
        imgSizeH: 224,
        imgSizeW: 224,
        saveModelDefault: true,
        offlineAugmentation: false,
        augmentationStrategyOverride: '',
    };

    return (
        <Container maxWidth="md">
            <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
                <Typography variant="h4" component="h1" gutterBottom>
                    Create New Experiment
                </Typography>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

                <Formik
                    initialValues={initialValues}
                    validationSchema={ExperimentSchema}
                    onSubmit={async (values) => {
                        setSubmitting(true);
                        setError('');
                        setSuccess('');

                        const payload = {
                            name: values.name,
                            datasetName: values.datasetName,
                            modelType: values.modelType,
                            methodsSequence: values.methodsSequence.map(m => ({
                                methodName: m.method_name,
                                // Convert params string back to object if it was stringified
                                params: (typeof m.params === 'string' && m.params.trim() !== "") ? JSON.parse(m.params) : (m.params || {}),
                                ...(m.save_model !== undefined && {save_model: m.save_model}), // Only include if defined
                                ...(m.save_best_model !== undefined && {save_best_model: m.save_best_model}),
                                ...(m.plot_level !== undefined && {plot_level: m.plot_level}),
                            })),
                            imgSizeH: values.imgSizeH || undefined, // Send undefined if empty for Python to use defaults
                            imgSizeW: values.imgSizeW || undefined,
                            saveModelDefault: values.saveModelDefault,
                            offlineAugmentation: values.offlineAugmentation,
                            augmentationStrategyOverride: values.augmentationStrategyOverride || undefined,
                        };

                        try {
                            const createdExperiment = await experimentService.createExperiment(payload);
                            setSuccess(`Experiment "${createdExperiment.name}" (ID: ${createdExperiment.experimentRunId}) submitted successfully!`);
                            // navigate(`/experiments/${createdExperiment.experimentRunId}`); // Optionally navigate
                            setTimeout(() => navigate('/experiments'), 2000);
                        } catch (err) {
                            setError(err.response?.data?.detail || err.message || 'Failed to create experiment.');
                        } finally {
                            setSubmitting(false);
                        }
                    }}
                >
                    {({ values, errors, touched, handleChange, handleBlur, setFieldValue }) => (
                        <Form>
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <Field as={TextField} name="name" label="Experiment Name" fullWidth required error={touched.name && Boolean(errors.name)} helperText={touched.name && errors.name} />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <FormControl fullWidth required error={touched.datasetName && Boolean(errors.datasetName)}>
                                        <InputLabel>Dataset</InputLabel>
                                        <Field as={Select} name="datasetName" label="Dataset">
                                            {DATASET_NAMES.map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)}
                                        </Field>
                                        {touched.datasetName && errors.datasetName && <FormHelperText error>{errors.datasetName}</FormHelperText>}
                                    </FormControl>
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <FormControl fullWidth required error={touched.modelType && Boolean(errors.modelType)}>
                                        <InputLabel>Model Type</InputLabel>
                                        <Field as={Select} name="modelType" label="Model Type">
                                            {MODEL_TYPES.map(mt => <MenuItem key={mt.value} value={mt.value}>{mt.label}</MenuItem>)}
                                        </Field>
                                        {touched.modelType && errors.modelType && <FormHelperText error>{errors.modelType}</FormHelperText>}
                                    </FormControl>
                                </Grid>

                                <Grid item xs={12}><Divider sx={{my:1}}><Typography variant="overline">Pipeline Settings</Typography></Divider></Grid>
                                <Grid item xs={6} sm={3}>
                                    <Field as={TextField} name="imgSizeH" label="Img Height" type="number" fullWidth InputLabelProps={{ shrink: true }}/>
                                </Grid>
                                <Grid item xs={6} sm={3}>
                                    <Field as={TextField} name="imgSizeW" label="Img Width" type="number" fullWidth InputLabelProps={{ shrink: true }}/>
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <FormControlLabel
                                        control={<Field as={Switch} type="checkbox" name="offlineAugmentation" />}
                                        label="Use Offline Augmented Data"
                                    />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <FormControl fullWidth error={touched.augmentationStrategyOverride && Boolean(errors.augmentationStrategyOverride)}>
                                        <InputLabel>Augmentation Override (Optional)</InputLabel>
                                        <Field as={Select} name="augmentationStrategyOverride" label="Augmentation Override (Optional)">
                                            <MenuItem value=""><em>(Pipeline Default)</em></MenuItem>
                                            {AVAILABLE_AUG_STRATEGIES.map(aug => <MenuItem key={aug.value} value={aug.value}>{aug.label}</MenuItem>)}
                                        </Field>
                                        {touched.augmentationStrategyOverride && errors.augmentationStrategyOverride && <FormHelperText error>{errors.augmentationStrategyOverride}</FormHelperText>}
                                    </FormControl>
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <FormControlLabel
                                        control={<Field as={Switch} type="checkbox" name="saveModelDefault" />}
                                        label="Save Model by Default (if method supports)"
                                    />
                                </Grid>


                                <Grid item xs={12}><Divider sx={{my:1}}><Typography variant="overline">Methods Sequence</Typography></Divider></Grid>
                                <FieldArray name="methodsSequence">
                                    {({ push, remove }) => (
                                        <Grid item xs={12}>
                                            {values.methodsSequence.map((method, index) => (
                                                <Paper key={index} elevation={1} sx={{ p: 2, mb: 2 }}>
                                                    <Grid container spacing={1} alignItems="center">
                                                        <Grid item xs={12} sm={4}>
                                                            <FormControl fullWidth required error={touched.methodsSequence?.[index]?.method_name && Boolean(errors.methodsSequence?.[index]?.method_name)}>
                                                                <InputLabel>Method</InputLabel>
                                                                <Field
                                                                    as={Select}
                                                                    name={`methodsSequence[${index}].method_name`}
                                                                    label="Method"
                                                                    onChange={(e) => {
                                                                        const newMethodName = e.target.value;
                                                                        setFieldValue(`methodsSequence[${index}].method_name`, newMethodName);
                                                                        // Set default params for the new method, stringifying the params part
                                                                        const defaults = getMethodDefaultParams(newMethodName);
                                                                        setFieldValue(`methodsSequence[${index}].params`, JSON.stringify(defaults.params || {}, null, 2));
                                                                        if(defaults.save_model !== undefined) setFieldValue(`methodsSequence[${index}].save_model`, defaults.save_model);
                                                                        if(defaults.save_best_model !== undefined) setFieldValue(`methodsSequence[${index}].save_best_model`, defaults.save_best_model);
                                                                        if(defaults.plot_level !== undefined) setFieldValue(`methodsSequence[${index}].plot_level`, defaults.plot_level);
                                                                    }}
                                                                >
                                                                    {PIPELINE_METHODS.map(pm => <MenuItem key={pm.value} value={pm.value}>{pm.label}</MenuItem>)}
                                                                </Field>
                                                                <ErrorMessage name={`methodsSequence[${index}].method_name`} component="div" style={{ color: 'red', fontSize: '0.75rem' }} />
                                                            </FormControl>
                                                        </Grid>
                                                        <Grid item xs={12} sm={method.method_name === 'single_eval' ? 7 : 3}>
                                                            <Field
                                                                as={TextField}
                                                                name={`methodsSequence[${index}].params`}
                                                                label="Method Parameters (JSON)"
                                                                fullWidth
                                                                multiline
                                                                rows={2}
                                                                variant="outlined"
                                                                size="small"
                                                                InputLabelProps={{ shrink: true }}
                                                                error={touched.methodsSequence?.[index]?.params && Boolean(errors.methodsSequence?.[index]?.params)}
                                                                helperText={touched.methodsSequence?.[index]?.params && errors.methodsSequence?.[index]?.params ? errors.methodsSequence[index].params : "e.g., {\"max_epochs\": 20}"}
                                                            />
                                                        </Grid>
                                                        {/* Conditional fields for save_model / plot_level etc. */}
                                                        {(method.method_name === 'single_train' || method.method_name === 'non_nested_grid_search') && (
                                                            <Grid item xs={6} sm={2}>
                                                                <FormControlLabel
                                                                    control={<Field as={Switch} type="checkbox" name={`methodsSequence[${index}].${method.method_name === 'single_train' ? 'save_model' : 'save_best_model'}`} />}
                                                                    label="Save"
                                                                />
                                                            </Grid>
                                                        )}
                                                        {(method.method_name !== 'predict_images' && method.method_name !== 'load_model') && (
                                                            <Grid item xs={6} sm={2}>
                                                                <FormControl fullWidth size="small">
                                                                    <InputLabel>Plot</InputLabel>
                                                                    <Field as={Select} name={`methodsSequence[${index}].plot_level`} label="Plot">
                                                                        <MenuItem value={0}>None</MenuItem>
                                                                        <MenuItem value={1}>Save</MenuItem>
                                                                        <MenuItem value={2}>Save & Show</MenuItem>
                                                                    </Field>
                                                                </FormControl>
                                                            </Grid>
                                                        )}


                                                        <Grid item xs={12} sm={1}>
                                                            <IconButton onClick={() => remove(index)} disabled={values.methodsSequence.length <= 1}>
                                                                <RemoveCircleOutlineIcon />
                                                            </IconButton>
                                                        </Grid>
                                                    </Grid>
                                                </Paper>
                                            ))}
                                            <Button
                                                type="button"
                                                startIcon={<AddIcon />}
                                                onClick={() => push({ method_name: PIPELINE_METHODS[0].value, ...getMethodDefaultParams(PIPELINE_METHODS[0].value) })}
                                                sx={{ mt: 1 }}
                                            >
                                                Add Method Step
                                            </Button>
                                        </Grid>
                                    )}
                                </FieldArray>

                                <Grid item xs={12} sx={{ mt: 2 }}>
                                    <Button type="submit" variant="contained" color="primary" disabled={submitting}>
                                        {submitting ? <CircularProgress size={24} /> : 'Create and Run Experiment'}
                                    </Button>
                                </Grid>
                            </Grid>
                        </Form>
                    )}
                </Formik>
            </Paper>
        </Container>
    );
};

export default CreateExperimentPage;