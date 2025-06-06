import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Formik, Form, Field, FieldArray, ErrorMessage, useFormikContext } from 'formik';
import * as Yup from 'yup';
import {
    Container, Typography, Paper, Box, Button, Grid, TextField, Select, MenuItem, InputLabel, FormControl,
    FormHelperText, CircularProgress, Alert, IconButton, Switch, FormControlLabel, Divider,
    ToggleButtonGroup, ToggleButton, Tooltip
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import CodeIcon from '@mui/icons-material/Code';
import ListAltIcon from '@mui/icons-material/ListAlt';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

import experimentService from '../services/experimentService';
// Assuming experimentConfig.js is in the same directory or adjust path
import { EXPERIMENT_MODES, METHOD_DEFAULTS, PRESET_SEQUENCES, MODEL_TYPES, DATASET_NAMES, AVAILABLE_AUG_STRATEGIES, PIPELINE_METHODS } from './experimentConfig';
import ParamsInfoModal from '../components/Modals/ParamsInfoModal'; // Assuming modal is in components/Modals

// Debounce helper (keep as is)
const debounce = (func, delay) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
};

// KeyValueSchema and ParamsSchema (keep as is)
const KeyValueSchema = Yup.object().shape({
    keyName: Yup.string().required('Key is required').matches(/^[a-zA-Z0-9_\[\].]+$/, 'Key can only contain letters, numbers, underscores, brackets, and dots.'),
    keyValue: Yup.string().required('Value is required'),
});

const ParamsSchema = Yup.lazy(value => {
    if (typeof value === 'string') {
        return Yup.string().test('is-json', 'Must be valid JSON', val => {
            if (!val || val.trim() === '{}' || val.trim() === '') return true;
            try { JSON.parse(val); return true; } catch (e) { return false; }
        });
    } else if (Array.isArray(value)) {
        return Yup.array().of(KeyValueSchema);
    }
    return Yup.object();
});

const ExperimentSchema = Yup.object().shape({
    name: Yup.string().required('Experiment name is required').max(255),
    datasetName: Yup.string().required('Dataset is required'),
    modelType: Yup.string().required('Model type is required'),
    experimentMode: Yup.string().required('Experiment mode is required'),
    methodsSequence: Yup.array()
        .of(
            Yup.object().shape({
                method_name: Yup.string().required('Method is required'),
                paramsEditorMode: Yup.string().oneOf(['kv', 'json']).required(),
                params: ParamsSchema,
                save_model: Yup.boolean().optional(),
                save_best_model: Yup.boolean().optional(),
                plot_level: Yup.number().min(0).max(2).integer().optional(),
                results_detail_level: Yup.number().min(0).max(3).integer().optional(),
                cv: Yup.number().integer().min(2).optional().nullable(), // Allow nullable for conditional logic
                outer_cv: Yup.number().integer().min(2).optional().nullable(),
                inner_cv: Yup.number().integer().min(2).optional().nullable(),
                scoring: Yup.string().optional().nullable(),
                method_search_type: Yup.string().oneOf(['grid', 'random']).optional().nullable(),
                n_iter: Yup.number().integer().min(1).optional().nullable(),
                evaluate_on: Yup.string().oneOf(['full', 'test']).optional().nullable(),
                internal_val_split_ratio: Yup.number().min(0.01).max(0.99).optional().nullable(),
                use_best_params_from_step: Yup.number().integer().min(0).optional().nullable(),
            })
        )
        .min(1, 'At least one method is required')
        .required('Method sequence is required'),
    imgSizeH: Yup.number().integer().positive().nullable().transform(value => (isNaN(value) || value === null || value === '' ? null : Number(value))),
    imgSizeW: Yup.number().integer().positive().nullable().transform(value => (isNaN(value) || value === null || value === '' ? null : Number(value))),
    offlineAugmentation: Yup.boolean(),
    augmentationStrategyOverride: Yup.string().nullable(),
});

// kvArrayToJsonString and jsonStringToKvArray (keep as is)
const kvArrayToJsonString = (kvArray) => {
    if (!Array.isArray(kvArray)) return '{}';
    const obj = {};
    kvArray.forEach(pair => {
        if (pair.keyName && pair.keyName.trim() !== "") {
            try { obj[pair.keyName.trim()] = JSON.parse(pair.keyValue); }
            catch (e) { obj[pair.keyName.trim()] = pair.keyValue; }
        }
    });
    return JSON.stringify(obj, null, 2);
};
const jsonStringToKvArray = (jsonString) => {
    try {
        const obj = JSON.parse(jsonString);
        if (typeof obj !== 'object' || obj === null) return [{ keyName: '', keyValue: '' }]; // Handle non-object JSON
        return Object.entries(obj).map(([keyName, value]) => ({
            keyName,
            keyValue: typeof value === 'string' ? value : JSON.stringify(value),
        }));
    } catch (e) { return [{ keyName: '', keyValue: '' }]; }
};


// CustomModeDetector (keep as is)
const CustomModeDetector = ({ initialSequence }) => {
    const { values, setFieldValue } = useFormikContext();
    const debouncedCheck = useCallback(
        debounce(() => {
            if (values.experimentMode !== 'custom') {
                const currentPresetSequence = PRESET_SEQUENCES[values.experimentMode] || [];
                let isDifferent = values.methodsSequence.length !== currentPresetSequence.length;
                if (!isDifferent) {
                    for (let i = 0; i < values.methodsSequence.length; i++) {
                        if (values.methodsSequence[i].method_name !== currentPresetSequence[i]?.method_name) {
                            isDifferent = true; break;
                        }
                    }
                }
                if (isDifferent) { setFieldValue('experimentMode', 'custom'); }
            }
        }, 500),
        [values.experimentMode, values.methodsSequence, setFieldValue]
    );
    useEffect(() => { debouncedCheck(); }, [values.methodsSequence, debouncedCheck]);
    return null;
};

// MethodStepCard Component (for memoization - keep as is or refine as needed)
const MethodStepCard = React.memo(({ method, index, values, errors, touched, handleChange, setFieldValue, remove }) => {
    const currentMethodName = method.method_name;
    const methodErrors = errors.methodsSequence?.[index];
    const methodTouched = touched.methodsSequence?.[index];

    // For Info Modal specific to this step
    const [stepInfoModalOpen, setStepInfoModalOpen] = useState(false);


    return (
        <Paper elevation={2} sx={{ p: 2, mb: 3, borderLeft: methodErrors ? '3px solid red' : '3px solid transparent' }}>
            <Grid container spacing={2} alignItems="flex-start">
                {/* Column 1: Method Type & Specific Controls */}
                <Grid item xs={12} md={4}>
                    <FormControl fullWidth required error={methodTouched?.method_name && Boolean(methodErrors?.method_name)} sx={{mb:1.5}}>
                        <InputLabel shrink>Method Type</InputLabel>
                        <Field
                            as={Select} name={`methodsSequence[${index}].method_name`} label="Method Type"
                            onChange={(e) => {
                                const newMethodName = e.target.value;
                                handleChange(e); // Update method_name first
                                const defaults = METHOD_DEFAULTS[newMethodName] || METHOD_DEFAULTS.single_train; // Fallback

                                // Set params and editor mode
                                setFieldValue(`methodsSequence[${index}].paramsEditorMode`, 'json');
                                setFieldValue(`methodsSequence[${index}].params`, JSON.stringify(defaults.params || {}, null, 2));

                                // Set other specific defaults or clear them
                                const allOptionalKeys = ['save_model', 'save_best_model', 'plot_level', 'results_detail_level', 'cv', 'outer_cv', 'inner_cv', 'scoring', 'method_search_type', 'n_iter', 'evaluate_on', 'internal_val_split_ratio', 'use_best_params_from_step'];
                                allOptionalKeys.forEach(key => {
                                    setFieldValue(`methodsSequence[${index}].${key}`, defaults[key] !== undefined ? defaults[key] : (key.endsWith('_level') ? 1 : undefined)); // Sensible default for levels
                                });
                            }}
                        >
                            {PIPELINE_METHODS.map(pm => <MenuItem key={pm.value} value={pm.value}>{pm.label}</MenuItem>)}
                        </Field>
                        <ErrorMessage name={`methodsSequence[${index}].method_name`} component={FormHelperText} error />
                    </FormControl>

                    {/* Method-Specific Controls */}
                    {(currentMethodName === 'single_train' || currentMethodName === 'non_nested_grid_search') && (
                        <FormControlLabel sx={{display:'flex', justifyContent: 'flex-start', mb:1}} control={
                            <Field as={Switch} type="checkbox" name={`methodsSequence[${index}].${currentMethodName === 'single_train' ? 'save_model' : 'save_best_model'}`} />}
                                          label="Save Model"
                        />
                    )}
                    { (currentMethodName !== 'load_model' && currentMethodName !== 'predict_images') &&
                        <FormControl fullWidth size="small" sx={{mb:1.5}}>
                            <InputLabel>Plot Level</InputLabel>
                            <Field as={Select} name={`methodsSequence[${index}].plot_level`} label="Plot Level">
                                <MenuItem value={0}>None</MenuItem><MenuItem value={1}>Save Only</MenuItem><MenuItem value={2}>Save & Show</MenuItem>
                            </Field>
                        </FormControl>
                    }
                    <FormControl fullWidth size="small" sx={{mb:1.5}}>
                        <InputLabel>Results Detail</InputLabel>
                        <Field as={Select} name={`methodsSequence[${index}].results_detail_level`} label="Results Detail">
                            <MenuItem value={0}>None (Summary Only)</MenuItem><MenuItem value={1}>Basic Summary</MenuItem><MenuItem value={2}>Detailed</MenuItem><MenuItem value={3}>Full (incl. Batch Data)</MenuItem>
                        </Field>
                    </FormControl>

                    {(currentMethodName === 'non_nested_grid_search' || currentMethodName === 'cv_model_evaluation' || currentMethodName === 'nested_grid_search') && (
                        <Field as={TextField} type="number" name={`methodsSequence[${index}].cv`} label="CV Folds" fullWidth size="small" sx={{mb:1.5}} InputLabelProps={{ shrink: true }} InputProps={{ inputProps: { min: 2 } }}/>
                    )}
                    {(currentMethodName === 'nested_grid_search') && (<>
                        <Field as={TextField} type="number" name={`methodsSequence[${index}].outer_cv`} label="Outer CV Folds" fullWidth size="small" sx={{mb:1.5}} InputLabelProps={{ shrink: true }} InputProps={{ inputProps: { min: 2 } }}/>
                        <Field as={TextField} type="number" name={`methodsSequence[${index}].inner_cv`} label="Inner CV Folds" fullWidth size="small" sx={{mb:1.5}} InputLabelProps={{ shrink: true }} InputProps={{ inputProps: { min: 2 } }}/>
                    </>)}
                    {(currentMethodName === 'non_nested_grid_search' || currentMethodName === 'nested_grid_search') && (<>
                        <Field as={TextField} name={`methodsSequence[${index}].scoring`} label="Scoring Metric" fullWidth size="small" sx={{mb:1.5}} InputLabelProps={{ shrink: true }}/>
                        <FormControl fullWidth size="small" sx={{mb:1.5}}> <InputLabel>Search Type</InputLabel>
                            <Field as={Select} name={`methodsSequence[${index}].method_search_type`} label="Search Type">
                                <MenuItem value="grid">Grid Search</MenuItem><MenuItem value="random">Random Search</MenuItem>
                            </Field>
                        </FormControl>
                        {method.method_search_type === 'random' && <Field as={TextField} type="number" name={`methodsSequence[${index}].n_iter`} label="Num Iterations (Random)" fullWidth size="small" sx={{mb:1.5}} InputLabelProps={{ shrink: true }} InputProps={{ inputProps: { min: 1 } }}/>}
                    </>)}
                    {(currentMethodName === 'cv_model_evaluation') && (
                        <FormControl fullWidth size="small" sx={{mb:1.5}}> <InputLabel>Evaluate On</InputLabel>
                            <Field as={Select} name={`methodsSequence[${index}].evaluate_on`} label="Evaluate On">
                                <MenuItem value="full">Full Dataset</MenuItem><MenuItem value="test">Test Set Only</MenuItem>
                            </Field>
                        </FormControl>
                    )}
                    {(currentMethodName === 'single_train' || currentMethodName === 'non_nested_grid_search' || currentMethodName === 'cv_model_evaluation') && (
                        <Field as={TextField} type="number" name={`methodsSequence[${index}].internal_val_split_ratio`} label="Internal Val Split (0-1)" fullWidth size="small" sx={{mb:1.5}} InputLabelProps={{ shrink: true }} InputProps={{inputProps: {step: "0.01", min:"0.01", max:"0.99"}}}/>
                    )}
                    {index > 0 && values.methodsSequence[index-1].method_name === 'non_nested_grid_search' && (currentMethodName === 'single_eval' || currentMethodName === 'cv_model_evaluation') && (
                        <FormControlLabel sx={{display:'flex', justifyContent: 'flex-start'}} control={
                            <Field
                                type="checkbox" as={Switch}
                                name={`methodsSequence[${index}].use_best_params_from_step_checkbox`}
                                checked={values.methodsSequence[index].use_best_params_from_step === (index - 1)}
                                onChange={e => {
                                    setFieldValue(`methodsSequence[${index}].use_best_params_from_step`, e.target.checked ? index - 1 : undefined);
                                    // For Formik with Switch, sometimes need to manually set the checkbox state if not using its 'value' prop
                                    setFieldValue(`methodsSequence[${index}].use_best_params_from_step_checkbox`, e.target.checked);
                                }}
                            />}
                                          label={`Use best params from Step ${index}`}
                        />
                    )}
                </Grid>

                {/* Column 2: Params Editor */}
                <Grid item xs={12} md={7}>
                    <Box sx={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1, minHeight: 40 /* Ensure a minimum height for alignment */ }}>
                        <Typography variant="body2" color="textSecondary"> {/* Removed sx={{ lineHeight: 'normal' }} for now, rely on alignItems */}
                            Method Parameters Block (for 'params' key)
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Tooltip title="Parameter Info/Help">
                                {/* Make IconButton slightly larger to match ToggleButton touch area better */}
                                <IconButton size="medium" onClick={() => setStepInfoModalOpen(true)} sx={{ mr: 0.5, p: '6px' /* Adjust padding to visually center icon */ }}>
                                    <HelpOutlineIcon fontSize="medium" /> {/* Increased icon size */}
                                </IconButton>
                            </Tooltip>
                            <ToggleButtonGroup
                                value={method.paramsEditorMode}
                                exclusive
                                size="small" // ToggleButtonGroup size="small" affects button padding
                                onChange={(_, newMode) => {
                                    // ... (conversion logic remains the same)
                                    if (newMode !== null && newMode !== method.paramsEditorMode) {
                                        const currentParams = values.methodsSequence[index].params;
                                        let convertedParams;
                                        if (newMode === 'kv') {
                                            convertedParams = typeof currentParams === 'string' ? jsonStringToKvArray(currentParams) : (Array.isArray(currentParams) ? currentParams : [{ keyName: '', keyValue: '' }]);
                                        } else { // json
                                            convertedParams = Array.isArray(currentParams) ? kvArrayToJsonString(currentParams) : (typeof currentParams === 'string' ? currentParams : '{}');
                                        }
                                        setFieldValue(`methodsSequence[${index}].paramsEditorMode`, newMode);
                                        setFieldValue(`methodsSequence[${index}].params`, convertedParams);
                                    }
                                }}
                            >
                                <ToggleButton value="kv" aria-label="key-value editor" sx={{lineHeight: 1 /* Help with icon centering in button */}}>
                                    <Tooltip title="Key-Value Editor"><ListAltIcon fontSize="medium"/></Tooltip> {/* Increased icon size */}
                                </ToggleButton>
                                <ToggleButton value="json" aria-label="raw json editor" sx={{lineHeight: 1}}>
                                    <Tooltip title="Raw JSON Editor"><CodeIcon fontSize="medium"/></Tooltip> {/* Increased icon size */}
                                </ToggleButton>
                            </ToggleButtonGroup>
                        </Box>
                    </Box>

                    {method.paramsEditorMode === 'json' ? (
                        <Field as={TextField} name={`methodsSequence[${index}].params`} fullWidth multiline rows={Math.max(4, ((typeof method.params === 'string' ? method.params : '{}').match(/\n/g) || []).length + 1)} variant="outlined" InputLabelProps={{ shrink: true }} label="Raw JSON for 'params'" error={methodTouched?.params && Boolean(methodErrors?.params)} helperText={methodTouched?.params && methodErrors?.params ? String(methodErrors.params) : "Enter valid JSON or {} for empty."}/>
                    ) : (
                        <FieldArray name={`methodsSequence[${index}].params`}>
                            {({ push: kvPush, remove: kvRemove }) => (
                                <Box sx={{maxHeight: 250, overflowY: 'auto', pr:1}}> {/* Scrollable KV area */}
                                    {Array.isArray(method.params) && method.params.map((kvPair, kvIndex) => (
                                        <Grid container spacing={1} key={kvIndex} alignItems="center" sx={{ mb: 2, mt: kvIndex === 0 ? 1.5 : 0 }}> {/* Increased mb, added mt for first item if needed */}
                                            <Grid item xs={5}>
                                                <Field
                                                    as={TextField}
                                                    name={`methodsSequence[${index}].params[${kvIndex}].keyName`}
                                                    label="Key"
                                                    fullWidth
                                                    size="small"
                                                    variant="outlined"
                                                    // Ensure label is always visible and shrunk
                                                    InputLabelProps={{ shrink: true }}
                                                    // Add some padding if necessary, or adjust variant if 'filled' or 'standard' work better visually
                                                    // sx={{ '& .MuiInputLabel-root': { visibility: 'visible' } }} // Force label visibility if shrink isn't enough
                                                />
                                            </Grid>
                                            <Grid item xs={5}>
                                                <Field
                                                    as={TextField}
                                                    name={`methodsSequence[${index}].params[${kvIndex}].keyValue`}
                                                    label="Value (JSON string for array/obj)"
                                                    fullWidth
                                                    size="small"
                                                    variant="outlined"
                                                    InputLabelProps={{ shrink: true }} // Ensure label is shrunk
                                                />
                                            </Grid>
                                            <Grid item xs={2} sx={{ textAlign: 'right', pt: 0.5 }}>
                                                <IconButton onClick={() => kvRemove(kvIndex)} size="small">
                                                    <RemoveCircleOutlineIcon fontSize="small"/>
                                                </IconButton>
                                            </Grid>
                                            <Grid item xs={12} sx={{pl: '8px !important', pt:0, mt: -0.5 }}>
                                                <ErrorMessage name={`methodsSequence[${index}].params[${kvIndex}].keyName`} component={FormHelperText} error />
                                                <ErrorMessage name={`methodsSequence[${index}].params[${kvIndex}].keyValue`} component={FormHelperText} error />
                                            </Grid>
                                        </Grid>
                                    ))}
                                    <Button type="button" size="small" startIcon={<AddIcon />} onClick={() => kvPush({ keyName: '', keyValue: '' })} sx={{mt:0.5}}>Add Parameter</Button>
                                </Box>
                            )}
                        </FieldArray>
                    )}
                </Grid>

                {/* Column 3: Remove Button */}
                <Grid item xs={12} md={1} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                    <IconButton onClick={() => remove(index)} disabled={values.methodsSequence.length <= 1}>
                        <RemoveCircleOutlineIcon color={values.methodsSequence.length <= 1 ? "disabled" : "error"}/>
                    </IconButton>
                </Grid>
            </Grid>
            <ParamsInfoModal
                open={stepInfoModalOpen}
                onClose={() => setStepInfoModalOpen(false)}
                currentMethodName={currentMethodName}
                currentModelType={values.modelType}
            />
        </Paper>
    );
});


const CreateExperimentPage = () => {
    const navigate = useNavigate();
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [submitting, setSubmitting] = useState(false);

    // State for the general info modal (if needed, but step-specific is better)
    // const [generalInfoModalOpen, setGeneralInfoModalOpen] = useState(false);

    const [currentPresetSequenceForDetector, setCurrentPresetSequenceForDetector] = useState(
        PRESET_SEQUENCES[EXPERIMENT_MODES[0].value].map(m => ({
            ...m,
            paramsEditorMode: 'json',
            params: typeof m.params === 'object' ? JSON.stringify(m.params, null, 2) : m.params,
        }))
    );

    const initialValues = {
        name: '',
        datasetName: DATASET_NAMES[0] || '',
        modelType: MODEL_TYPES[0]?.value || '',
        experimentMode: EXPERIMENT_MODES[0].value,
        methodsSequence: PRESET_SEQUENCES[EXPERIMENT_MODES[0].value].map(m => ({
            ...m,
            paramsEditorMode: 'json',
            params: typeof m.params === 'object' ? JSON.stringify(m.params || {}, null, 2) : (m.params || '{}'),
            // Initialize all potential optional fields for consistent structure
            plot_level: m.plot_level !== undefined ? m.plot_level : 1,
            results_detail_level: m.results_detail_level !== undefined ? m.results_detail_level : 2,
            // Other fields will be undefined initially if not in METHOD_DEFAULTS for the specific method
        })),
        imgSizeH: 224,
        imgSizeW: 224,
        // saveModelDefault: true, // Removed global one
        offlineAugmentation: false,
        augmentationStrategyOverride: '',
    };


    return (
        <Container maxWidth="lg" sx={{pb: 5}}>
            <Paper elevation={3} sx={{ p: {xs: 2, md: 3}, mt: 3 }}>
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

                        const pythonPayloadMethods = values.methodsSequence.map(m => {
                            let finalParams = {};
                            if (m.paramsEditorMode === 'kv' && Array.isArray(m.params)) {
                                m.params.forEach(pair => {
                                    if (pair.keyName && pair.keyName.trim() !== "") {
                                        let parsedValue;
                                        try { parsedValue = JSON.parse(pair.keyValue); }
                                        catch (e) { parsedValue = pair.keyValue; } // Store as string if not valid JSON primitive/array/object

                                        // Handle comma-separated strings for grid search lists
                                        if (typeof parsedValue === 'string' && parsedValue.includes(',') && (m.method_name === 'non_nested_grid_search' || m.method_name === 'nested_grid_search')) {
                                            const potentialList = parsedValue.split(',').map(s => s.trim());
                                            // Basic check if it looks like a list of numbers or simple strings
                                            if (potentialList.every(item => !isNaN(parseFloat(item))) && potentialList.some(item => item.includes('.'))) {
                                                finalParams[pair.keyName.trim()] = potentialList.map(parseFloat);
                                            } else if (potentialList.every(item => /^-?\d+$/.test(item))) {
                                                finalParams[pair.keyName.trim()] = potentialList.map(Number);
                                            } else {
                                                finalParams[pair.keyName.trim()] = potentialList; // As list of strings
                                            }
                                        } else {
                                            finalParams[pair.keyName.trim()] = parsedValue;
                                        }
                                    }
                                });
                            } else if (typeof m.params === 'string') {
                                try { finalParams = m.params.trim() ? JSON.parse(m.params) : {}; }
                                catch (e) {
                                    setError(`Invalid JSON in params for method '${m.method_name}': ${e.message}. Please correct it.`);
                                    setSubmitting(false);
                                    throw new Error("Invalid JSON params for method: " + m.method_name);
                                }
                            } else {
                                finalParams = m.params || {}; // Should already be an object if not string/array
                            }

                            const methodPayload = {
                                methodName: m.method_name,
                                params: finalParams,
                            };

                            // Add optional top-level args for the Python method call
                            if (m.save_model !== undefined) methodPayload.save_model = m.save_model;
                            if (m.save_best_model !== undefined) methodPayload.save_best_model = m.save_best_model;
                            if (m.plot_level !== undefined) methodPayload.plot_level = m.plot_level;
                            if (m.results_detail_level !== undefined) methodPayload.results_detail_level = m.results_detail_level;
                            if (m.cv !== undefined && m.cv !== null && m.cv !== '') methodPayload.cv = Number(m.cv);
                            if (m.outer_cv !== undefined && m.outer_cv !== null && m.outer_cv !== '') methodPayload.outer_cv = Number(m.outer_cv);
                            if (m.inner_cv !== undefined && m.inner_cv !== null && m.inner_cv !== '') methodPayload.inner_cv = Number(m.inner_cv);
                            if (m.scoring) methodPayload.scoring = m.scoring;
                            if (m.method_search_type) methodPayload.method = m.method_search_type; // Python uses 'method'
                            if (m.n_iter !== undefined && m.n_iter !== null && m.n_iter !== '' && m.method_search_type === 'random') methodPayload.n_iter = Number(m.n_iter);
                            if (m.evaluate_on) methodPayload.evaluate_on = m.evaluate_on;
                            if (m.internal_val_split_ratio !== undefined && m.internal_val_split_ratio !== null && m.internal_val_split_ratio !== '') methodPayload.internal_val_split_ratio = Number(m.internal_val_split_ratio);
                            if (m.use_best_params_from_step !== undefined && m.use_best_params_from_step !== null) methodPayload.use_best_params_from_step = Number(m.use_best_params_from_step);

                            return methodPayload;
                        });

                        const finalPayload = {
                            name: values.name,
                            datasetName: values.datasetName,
                            modelType: values.modelType,
                            methodsSequence: pythonPayloadMethods,
                            imgSizeH: values.imgSizeH ? Number(values.imgSizeH) : undefined,
                            imgSizeW: values.imgSizeW ? Number(values.imgSizeW) : undefined,
                            // saveModelDefault: values.saveModelDefault, // Removed from UI, Python will use its default or method specific
                            offlineAugmentation: values.offlineAugmentation,
                            augmentationStrategyOverride: values.augmentationStrategyOverride || undefined,
                        };
                        // console.log("Final Payload to Java:", JSON.stringify(finalPayload, null, 2)); // For debugging

                        try {
                            const createdExperiment = await experimentService.createExperiment(finalPayload);
                            setSuccess(`Experiment "${createdExperiment.name}" (ID: ${createdExperiment.experimentRunId}) submitted!`);
                            setTimeout(() => navigate('/experiments'), 2500);
                        } catch (err) {
                            setError(err.response?.data?.detail || err.message || 'Failed to create experiment.');
                        } finally {
                            setSubmitting(false);
                        }
                    }}
                >
                    {({ values, errors, touched, handleChange, handleBlur, setFieldValue, isSubmitting }) => (
                        <Form>
                            <CustomModeDetector initialSequence={currentPresetSequenceForDetector} />
                            <Grid container spacing={3}>
                                <Grid item xs={12}> <Typography variant="h6">Basic Experiment Info</Typography> </Grid>
                                <Grid item xs={12}><Field as={TextField} name="name" label="Experiment Name" fullWidth required error={touched.name && Boolean(errors.name)} helperText={touched.name && errors.name} /></Grid>
                                <Grid item xs={12} sm={4}><FormControl fullWidth required error={touched.datasetName && Boolean(errors.datasetName)}><InputLabel>Dataset</InputLabel><Field as={Select} name="datasetName" label="Dataset">{DATASET_NAMES.map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)}</Field>{touched.datasetName && errors.datasetName && <FormHelperText error>{errors.datasetName}</FormHelperText>}</FormControl></Grid>
                                <Grid item xs={12} sm={4}><FormControl fullWidth required error={touched.modelType && Boolean(errors.modelType)}><InputLabel>Model Type</InputLabel><Field as={Select} name="modelType" label="Model Type">{MODEL_TYPES.map(mt => <MenuItem key={mt.value} value={mt.value}>{mt.label}</MenuItem>)}</Field>{touched.modelType && errors.modelType && <FormHelperText error>{errors.modelType}</FormHelperText>}</FormControl></Grid>
                                <Grid item xs={12} sm={4}><FormControl fullWidth required><InputLabel>Experiment Mode</InputLabel>
                                    <Field as={Select} name="experimentMode" label="Experiment Mode"
                                           onChange={(e) => {
                                               const mode = e.target.value;
                                               handleChange(e);
                                               if (mode !== 'custom') {
                                                   const presetSeq = PRESET_SEQUENCES[mode].map(m => ({
                                                       ...METHOD_DEFAULTS[m.method_name], // Start with full defaults for the method
                                                       ...m, // Override with preset specifics
                                                       paramsEditorMode: 'json',
                                                       params: JSON.stringify( (m.params || METHOD_DEFAULTS[m.method_name]?.params || {}), null, 2),
                                                   }));
                                                   setFieldValue('methodsSequence', presetSeq);
                                                   setCurrentPresetSequenceForDetector(presetSeq); // For custom mode detection
                                               }
                                           }}
                                    >{EXPERIMENT_MODES.map(em => <MenuItem key={em.value} value={em.value}>{em.label}</MenuItem>)}</Field>
                                </FormControl></Grid>

                                <Grid item xs={12} sx={{mt:1}}><Typography variant="h6">Global Pipeline Settings</Typography><Divider/></Grid>
                                <Grid item xs={6} sm={3}><Field as={TextField} name="imgSizeH" label="Img Height" type="number" fullWidth InputLabelProps={{ shrink: true }} /></Grid>
                                <Grid item xs={6} sm={3}><Field as={TextField} name="imgSizeW" label="Img Width" type="number" fullWidth InputLabelProps={{ shrink: true }} /></Grid>
                                <Grid item xs={12} sm={6} sx={{display:'flex', alignItems:'center'}}><FormControlLabel control={<Field as={Switch} type="checkbox" name="offlineAugmentation" />} label="Use Offline Augmented Data" /></Grid>
                                <Grid item xs={12} sm={6}> {/* Augmentation Override Grid item */}
                                    <FormControl fullWidth error={touched.augmentationStrategyOverride && Boolean(errors.augmentationStrategyOverride)}>
                                        <InputLabel id="augmentation-override-label">
                                            Augmentation Override (Optional)
                                        </InputLabel>
                                        <Field
                                            as={Select}
                                            name="augmentationStrategyOverride"
                                            labelId="augmentation-override-label"
                                            label="Augmentation Override (Optional)"

                                            sx={{
                                                '& .MuiSelect-select': { // Targets the inner div that displays the selected value
                                                    minWidth: '150px',    // Example: Force a minimum width for the display area
                                                                          // Adjust this value based on your longest expected label
                                                    overflow: 'hidden',
                                                    textOverflow: 'ellipsis', // Ensure ellipsis if it still overflows minWidth
                                                    whiteSpace: 'nowrap',    // Typically for select display, you want it on one line
                                                },
                                                // Ensure the outer Select box itself has enough room
                                                // minWidth: 200, // If the above doesn't give enough space for the ellipsis
                                            }}

                                            // Ensure the Select component itself can accommodate wider text
                                            // by default, or by setting an explicit width or sx prop on the Select itself if needed.
                                            // The MenuProps primarily control the dropdown menu.
                                            MenuProps={{
                                                PaperProps: {
                                                    style: {
                                                        maxHeight: 300, // Increased max height for dropdown
                                                        width: 'auto',   // Allow paper to be wider than select if needed
                                                        // or set a specific minWidth: e.g., 250 or 'fit-content'
                                                    },
                                                },
                                            }}
                                            renderValue={(selectedValue) => {
                                                if (!selectedValue) {
                                                    return <em style={{ opacity: 0.7 }}>(Pipeline Default)</em>;
                                                }
                                                const selectedStrategy = AVAILABLE_AUG_STRATEGIES.find(s => s.value === selectedValue);
                                                // Render the selected value; MUI's Select should handle ellipsis if it overflows its own bounds
                                                return selectedStrategy ? selectedStrategy.label : selectedValue;
                                            }}
                                            // Add sx to the Select itself if its default width is too constrained
                                            // sx={{ minWidth: 200 /* Example: ensure a minimum width for the select box */ }}
                                        >
                                            <MenuItem value="">
                                                <em style={{ opacity: 0.7 }}>(Pipeline Default)</em>
                                            </MenuItem>
                                            {AVAILABLE_AUG_STRATEGIES.map(aug => (
                                                <MenuItem
                                                    key={aug.value}
                                                    value={aug.value}
                                                    // sx prop on MenuItem to allow text wrapping and ensure full width is utilized
                                                    sx={{
                                                        whiteSpace: 'normal', // Allow text to wrap
                                                        // If you want to ensure items take more width in the dropdown:
                                                        minWidth: 'fit-content', // or a specific pixel value
                                                        // Or ensure the Paper of MenuProps is wide enough
                                                    }}
                                                >
                                                    {/* No need for Tooltip here if whiteSpace: 'normal' works,
                        but keep if you prefer hover for very long ones. */}
                                                    {/* <Tooltip title={aug.label} placement="right-start"> */}
                                                    <Typography variant="body2" component="div" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'normal' }}>
                                                        {aug.label}
                                                    </Typography>
                                                    {/* </Tooltip> */}
                                                </MenuItem>
                                            ))}
                                        </Field>
                                        {touched.augmentationStrategyOverride && errors.augmentationStrategyOverride && <FormHelperText error>{errors.augmentationStrategyOverride}</FormHelperText>}
                                    </FormControl>
                                </Grid>


                                <Grid item xs={12} sx={{mt:1}}><Typography variant="h6">Methods Sequence</Typography><Divider /></Grid>
                                <FieldArray name="methodsSequence">
                                    {({ push, remove }) => (
                                        <Grid item xs={12}>
                                            {values.methodsSequence.map((methodItem, index) => (
                                                <MethodStepCard
                                                    key={`method-${index}-${methodItem.method_name}`} // More unique key
                                                    method={methodItem} index={index} values={values} errors={errors}
                                                    touched={touched} handleChange={handleChange}
                                                    setFieldValue={setFieldValue} remove={remove}
                                                />
                                            ))}
                                            <Button type="button" startIcon={<AddIcon />} onClick={() => {
                                                const defaultNewMethod = METHOD_DEFAULTS.single_train; // Sensible default
                                                push({
                                                    ...defaultNewMethod,
                                                    paramsEditorMode: 'json',
                                                    params: JSON.stringify(defaultNewMethod.params || {}, null, 2)
                                                });
                                            }} sx={{ mt: 1 }}>Add Method Step</Button>
                                        </Grid>
                                    )}
                                </FieldArray>

                                <Grid item xs={12} sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}> {/* Centered Button */}
                                    <Button type="submit" variant="contained" color="primary" size="large" disabled={isSubmitting}>
                                        {isSubmitting ? <CircularProgress size={24} /> : 'Create and Run Experiment'}
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