import React, {useState, useEffect, useCallback, useRef} from 'react';
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
import {
    EXPERIMENT_MODES,
    METHOD_DEFAULTS,
    PRESET_SEQUENCES,
    MODEL_TYPES,
    DATASET_NAMES,
    AVAILABLE_AUG_STRATEGIES,
    PIPELINE_METHODS,
    PRESET_SEQUENCES_INITIALIZED
} from './experimentConfig';
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
                val_split_ratio: Yup.number().min(0.01).max(0.99).optional().nullable(),
                use_best_params_from_step: Yup.number().integer().min(0).optional().nullable(),
            })
        )
        .min(1, 'At least one method is required')
        .required('Method sequence is required'),
    imgSizeH: Yup.number().integer().positive().nullable().transform(value => (isNaN(value) || value === null || value === '' ? null : Number(value))),
    imgSizeW: Yup.number().integer().positive().nullable().transform(value => (isNaN(value) || value === null || value === '' ? null : Number(value))),
    offlineAugmentation: Yup.boolean(),
    augmentationStrategyOverride: Yup.string().nullable(),
    test_split_ratio_if_flat: Yup.number().min(0.01).max(0.99).nullable()
        .transform(value => (isNaN(value) || value === null || value === '' ? null : Number(value))),
    force_flat_for_fixed_cv: Yup.boolean(),
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
const MethodStepCard = React.memo(({ method, index, values, errors, touched, handleChange, setFieldValue, remove, previousMethodName }) => {
    const currentMethodName = method.method_name;
    const methodErrors = errors.methodsSequence?.[index];
    const methodTouched = touched.methodsSequence?.[index];

    // For Info Modal specific to this step
    const [stepInfoModalOpen, setStepInfoModalOpen] = useState(false);

    const showUseBestParamsControl =
        index > 0 &&
        previousMethodName === 'non_nested_grid_search' &&
        currentMethodName === 'cv_model_evaluation'; // ONLY for cv_model_evaluation

    const useBestParamsIsActive = (method.use_best_params_from_step !== undefined &&
            method.use_best_params_from_step !== null &&
            String(method.use_best_params_from_step).trim() !== '') &&
        currentMethodName === 'cv_model_evaluation'; // Only relevant for cv_model_evaluation

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
                                const allOptionalKeys = ['save_model', 'save_best_model', 'plot_level', 'results_detail_level', 'cv', 'outer_cv', 'inner_cv', 'scoring', 'method_search_type', 'n_iter', 'evaluate_on', 'val_split_ratio', 'use_best_params_from_step'];
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
                        <Field as={TextField} type="number" name={`methodsSequence[${index}].val_split_ratio`} label="Internal Val Split (0-1)" fullWidth size="small" sx={{mb:1.5}} InputLabelProps={{ shrink: true }} InputProps={{inputProps: {step: "0.01", min:"0.01", max:"0.99"}}}/>
                    )}
                    {/* Only show "Use best params" for cv_model_evaluation if applicable */}
                    {showUseBestParamsControl && (
                        <FormControl fullWidth size="small" sx={{mb:1.5}}>
                            <Field
                                as={TextField}
                                type="number"
                                name={`methodsSequence[${index}].use_best_params_from_step`}
                                label="Use Best Params from Step #"
                                placeholder={`0-${index - 1} (optional)`}
                                InputLabelProps={{ shrink: true }}
                                InputProps={{ inputProps: { min: 0, max: index - 1 } }}
                                helperText={ index > 0 ? `Index of previous step (0 to ${index - 1}).` : "No previous step."}
                                // onChange logic in Formik's onSubmit will handle converting empty to undefined
                            />
                            <ErrorMessage name={`methodsSequence[${index}].use_best_params_from_step`} component={FormHelperText} error />
                        </FormControl>
                    )}
                </Grid>

                {/* Column 2: Params Editor */}
                {currentMethodName !== 'single_eval' ? (
                    <Grid item xs={12} md={7}>
                        <Box sx={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1}}>
                            <Typography variant="body2" color="textSecondary">
                                {currentMethodName.includes('search') ? "Skorch HP Search Space (for 'param_grid')" : "Skorch Hyperparameters (for 'params')"}
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

                        {/* Add helper text if "Use best params" is active */}
                        {useBestParamsIsActive && currentMethodName === 'cv_model_evaluation' && (
                            <Alert severity="info" variant="outlined" icon={false} sx={{ fontSize: '0.75rem', p:1, mb: 1 }}>
                                Note: Parameters defined below will override or merge with the best parameters from Step {method.use_best_params_from_step}.
                            </Alert>
                        )}

                        {method.paramsEditorMode === 'json' ? (
                            <Field as={TextField}
                                   name={`methodsSequence[${index}].params`}
                                   fullWidth multiline rows={Math.max(4, ((typeof method.params === 'string' ? method.params : '{}').match(/\n/g) || []).length + 1)} variant="outlined" InputLabelProps={{ shrink: true }} label="Raw JSON for 'params'" error={methodTouched?.params && Boolean(methodErrors?.params)} helperText={methodTouched?.params && methodErrors?.params ? String(methodErrors.params) : "Enter valid JSON or {} for empty."}/>
                        ) : (
                            <FieldArray name={`methodsSequence[${index}].params`}>
                                {({ push: kvPush, remove: kvRemove }) => (
                                    <Box sx={{opacity: 1, pointerEvents: 'auto' }}> {/* Scrollable KV area */}
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
                                        <Button type="button" size="small"
                                                startIcon={<AddIcon />}
                                                onClick={() => kvPush({ keyName: '', keyValue: '' })} sx={{mt:0.5}}>Add Parameter</Button>
                                    </Box>
                                )}
                            </FieldArray>
                        )}
                    </Grid>
                ) : ( // For single_eval
                    <Grid item xs={12} md={7} sx={{display:'flex', alignItems:'center', justifyContent:'center', height:'100%'}}>
                        <Typography variant="caption" color="text.secondary" sx={{textAlign: 'center', p:2}}>
                            This step evaluates the model currently in memory. <br/>
                            {showUseBestParamsControl && "If 'Use Best Params from Step' is set, it implies the model from that step's tuning is used."}
                        </Typography>
                    </Grid>
                )}

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

    const [currentPresetSequenceForDetector, setCurrentPresetSequenceForDetector] = useState(
        PRESET_SEQUENCES[EXPERIMENT_MODES[0].value].map(m => ({
            ...m,
            paramsEditorMode: 'json',
            params: typeof m.params === 'object' ? JSON.stringify(m.params, null, 2) : m.params,
        }))
    );

    const pageTopRef = useRef(null);

    const scrollToTopMessages = () => {
        if (pageTopRef.current) {
            pageTopRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };

    const initialValues = {
        name: '',
        datasetName: DATASET_NAMES[0] || '',
        modelType: MODEL_TYPES[0]?.value || '',
        experimentMode: EXPERIMENT_MODES[0].value,
        methodsSequence: PRESET_SEQUENCES_INITIALIZED[EXPERIMENT_MODES[0].value],
        imgSizeH: 224,
        imgSizeW: 224,
        // saveModelDefault: true, // Removed global one
        offlineAugmentation: false,
        augmentationStrategyOverride: '',
        test_split_ratio_if_flat: 0.2, // Default value
        force_flat_for_fixed_cv: false,  // Default value
    };


    return (
        <>
            <div ref={pageTopRef} />
            <Container maxWidth="lg" sx={{pb: 5}}>
                {/* Add the ref to an element at the top, e.g., the Paper or a Box before alerts */}
                <Paper elevation={3} sx={{ p: {xs: 2, md: 3}, mt: 3 }}>
                    <Typography variant="h4" component="h1" gutterBottom>
                        Create New Experiment
                    </Typography>
                    {/* Error and Success Alerts are now effectively at the top */}
                    {error && <Alert severity="error" onClose={() => setError('')} sx={{ mb: 2 }}>{error}</Alert>}
                    {success && <Alert severity="success" onClose={() => setSuccess('')} sx={{ mb: 2 }}>{success}</Alert>}

                    <Formik
                        initialValues={initialValues}
                        validationSchema={ExperimentSchema}
                        onSubmit={async (values, { setSubmitting, resetForm }) => {
                            // setSubmitting(true) is handled by Formik automatically when onSubmit starts
                            setError('');
                            setSuccess('');
                            scrollToTopMessages(); // Scroll up when submission starts (or before API call)

                            // This is the payload structure for the Java DTO: ExperimentCreateRequest
                            // It will contain a list of PythonExperimentMethodParamsDTO objects
                            const javaMethodsSequence = values.methodsSequence.map((uiMethod, index) => {
                                let processedParamsObject = {}; // This will hold Skorch HPs or the param_grid content
                                if (uiMethod.paramsEditorMode === 'kv' && Array.isArray(uiMethod.params)) {
                                    uiMethod.params.forEach(pair => {
                                        if (pair.keyName && pair.keyName.trim() !== "") {
                                            let parsedValue;
                                            try {
                                                // Attempt to parse value as JSON (e.g., for numbers, booleans, arrays from strings)
                                                parsedValue = JSON.parse(pair.keyValue);
                                            } catch (e) {
                                                parsedValue = pair.keyValue; // Store as string if not valid JSON primitive/array/object
                                            }
                                            // Heuristic for comma-separated lists for search grids
                                            if (typeof parsedValue === 'string' && parsedValue.includes(',') &&
                                                (uiMethod.method_name === 'non_nested_grid_search' || uiMethod.method_name === 'nested_grid_search')) {
                                                const potentialList = parsedValue.split(',').map(s => s.trim());
                                                if (potentialList.every(item => !isNaN(parseFloat(item))) && potentialList.some(item => item.includes('.'))) {
                                                    processedParamsObject[pair.keyName.trim()] = potentialList.map(parseFloat);
                                                } else if (potentialList.every(item => /^-?\d+$/.test(item))) {
                                                    processedParamsObject[pair.keyName.trim()] = potentialList.map(Number);
                                                } else {
                                                    processedParamsObject[pair.keyName.trim()] = potentialList;
                                                }
                                            } else {
                                                processedParamsObject[pair.keyName.trim()] = parsedValue;
                                            }
                                        }
                                    });
                                } else if (typeof uiMethod.params === 'string') { // Raw JSON editor
                                    try {
                                        processedParamsObject = uiMethod.params.trim() ? JSON.parse(uiMethod.params) : {};
                                    } catch (e) {
                                        setError(`Invalid JSON in parameters for method '${uiMethod.method_name}': ${e.message}. Please correct it before submitting.`);
                                        setSubmitting(false);
                                        // Throw an error to stop Formik submission
                                        throw new Error("Invalid JSON params for method: " + uiMethod.method_name);
                                    }
                                } else { // Should ideally not happen if editor mode logic is correct
                                    processedParamsObject = uiMethod.params || {};
                                }

                                // This object maps to PythonExperimentMethodParamsDTO in Java
                                const methodPayloadForJava = { method_name: uiMethod.method_name };

                                if (uiMethod.method_name === 'non_nested_grid_search' || uiMethod.method_name === 'nested_grid_search') {
                                    methodPayloadForJava.param_grid = processedParamsObject; // The editable block IS the param_grid
                                    methodPayloadForJava.params = undefined; // Explicitly set params to empty or undefined if not used
                                } else if (uiMethod.method_name === 'single_train' || uiMethod.method_name === 'cv_model_evaluation') {
                                    methodPayloadForJava.params = processedParamsObject; // For single_train, cv_model_evaluation
                                    methodPayloadForJava.param_grid = undefined; // Or null, ensure it's not sent if not applicable
                                }

                                // Add other top-level method-specific controls (React state -> Java DTO camelCase)
                                if (uiMethod.save_model !== undefined) methodPayloadForJava.save_model = uiMethod.save_model;
                                if (uiMethod.save_best_model !== undefined) methodPayloadForJava.save_best_model = uiMethod.save_best_model;
                                if (uiMethod.plot_level !== undefined) methodPayloadForJava.plot_level = uiMethod.plot_level;
                                if (uiMethod.results_detail_level !== undefined) methodPayloadForJava.results_detail_level = uiMethod.results_detail_level;

                                // Ensure numeric values are Numbers or undefined/null
                                const toNumberOrUndefined = (val) => (val !== undefined && val !== null && String(val).trim() !== '') ? Number(val) : undefined;

                                methodPayloadForJava.cv = toNumberOrUndefined(uiMethod.cv);
                                methodPayloadForJava.outer_cv = toNumberOrUndefined(uiMethod.outer_cv);
                                methodPayloadForJava.inner_cv = toNumberOrUndefined(uiMethod.inner_cv);

                                if (uiMethod.scoring && String(uiMethod.scoring).trim() !== '') methodPayloadForJava.scoring = uiMethod.scoring;
                                if (uiMethod.method_search_type && String(uiMethod.method_search_type).trim() !== '') methodPayloadForJava.method_search_type = uiMethod.method_search_type;

                                if (uiMethod.method_search_type === 'random') {
                                    methodPayloadForJava.n_iter = toNumberOrUndefined(uiMethod.n_iter);
                                } else {
                                    methodPayloadForJava.n_iter = undefined; // Don't send if not random search
                                }

                                if (uiMethod.evaluate_on && String(uiMethod.evaluate_on).trim() !== '') methodPayloadForJava.evaluate_on = uiMethod.evaluate_on;
                                methodPayloadForJava.val_split_ratio = toNumberOrUndefined(uiMethod.val_split_ratio);

                                // 'use_best_params_from_step' is now just a number or undefined
                                const stepIndexToUse = toNumberOrUndefined(uiMethod.use_best_params_from_step);
                                if (stepIndexToUse !== undefined && stepIndexToUse >= 0 && stepIndexToUse < index) {
                                    methodPayloadForJava.use_best_params_from_step = stepIndexToUse;
                                    // No need to clear params here; Python will handle overriding/merging.
                                } else {
                                    methodPayloadForJava.use_best_params_from_step = undefined;
                                }

                                return methodPayloadForJava;
                            });

                            const finalPayloadForJava = { // This is ExperimentCreateRequest
                                name: values.name,
                                dataset_name: values.datasetName,
                                model_type: values.modelType,
                                methods_sequence: javaMethodsSequence,
                                img_size_h: values.imgSizeH ? Number(values.imgSizeH) : undefined,
                                img_size_w: values.imgSizeW ? Number(values.imgSizeW) : undefined,
                                offline_augmentation: values.offlineAugmentation,
                                augmentation_strategy_override: values.augmentationStrategyOverride || undefined,
                                test_split_ratio_if_flat: values.test_split_ratio_if_flat ? Number(values.test_split_ratio_if_flat) : undefined,
                                random_seed: values.random_seed ? Number(values.random_seed) : undefined,
                                force_flat_for_fixed_cv: values.force_flat_for_fixed_cv,
                            };

                            // console.log("Final Payload to Java:", JSON.stringify(finalPayloadForJava, null, 2));

                            try {
                                const createdExperiment = await experimentService.createExperiment(finalPayloadForJava);
                                setSuccess(`Experiment "${createdExperiment.name}" (ID: ${createdExperiment.experimentRunId}) submitted successfully! Check the Experiments page for status.`);
                                scrollToTopMessages(); // Scroll up to show success message
                                resetForm(); // Optional: reset form after successful submission
                                // Keep button disabled after success for a short while or until navigation
                                // Formik's setSubmitting(false) will re-enable it if not handled.
                                // We can let it be re-enabled and the user can navigate away.
                                // Or, if navigating away:
                                setTimeout(() => {
                                    if (navigate) navigate('/experiments');
                                }, 2500); // Navigate after showing success
                            } catch (err) {
                                const errorMsg = err.response?.data?.detail || err.message || 'Failed to create experiment.';
                                setError(errorMsg);
                                scrollToTopMessages(); // Scroll up to show error message
                            } finally {
                                // setSubmitting(false) is handled by Formik automatically when onSubmit finishes
                                // (unless an unhandled promise rejection occurs inside onSubmit)
                                // If you want to ensure it's set:
                                if (typeof setSubmitting === 'function') {
                                    setSubmitting(false);
                                }
                            }
                        }}
                    >
                        {({ values, errors, touched, handleChange, handleBlur, setFieldValue, isSubmitting }) => (
                            <Form>
                                <CustomModeDetector initialSequence={currentPresetSequenceForDetector} />
                                <Grid container spacing={3}> {/* Outer container for all form elements */}
                                    {/* Section 1: Basic Experiment Info */}
                                    <Grid item xs={12}> {/* This Grid item spans full width for its content row */}
                                        <Typography variant="h6" gutterBottom>Basic Experiment Info</Typography>
                                        <Divider sx={{ mb: 2 }} />
                                        <Grid container spacing={2}> {/* Nested container for this section's fields */}
                                            <Grid item xs={12}>
                                                <Field as={TextField} name="name" label="Experiment Name" fullWidth required error={touched.name && Boolean(errors.name)} helperText={touched.name && errors.name} />
                                            </Grid>
                                            <Grid item xs={12} sm={4}>
                                                <FormControl fullWidth required error={touched.datasetName && Boolean(errors.datasetName)}>
                                                    <InputLabel>Dataset</InputLabel>
                                                    <Field as={Select} name="datasetName" label="Dataset">{DATASET_NAMES.map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)}</Field>
                                                    {touched.datasetName && errors.datasetName && <FormHelperText error>{errors.datasetName}</FormHelperText>}
                                                </FormControl>
                                            </Grid>
                                            <Grid item xs={12} sm={4}>
                                                <FormControl fullWidth required error={touched.modelType && Boolean(errors.modelType)}>
                                                    <InputLabel>Model Type</InputLabel>
                                                    <Field as={Select} name="modelType" label="Model Type">{MODEL_TYPES.map(mt => <MenuItem key={mt.value} value={mt.value}>{mt.label}</MenuItem>)}</Field>
                                                    {touched.modelType && errors.modelType && <FormHelperText error>{errors.modelType}</FormHelperText>}
                                                </FormControl>
                                            </Grid>
                                            <Grid item xs={12} sm={4}>
                                                <FormControl fullWidth required>
                                                    <InputLabel>Experiment Mode</InputLabel>
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
                                                </FormControl>
                                            </Grid>
                                        </Grid>
                                    </Grid>

                                    {/* Section 2: Global Pipeline Settings */}
                                    <Grid item xs={12} sx={{ mt: 3 }}> {/* This Grid item spans full width for this section */}
                                        <Typography variant="h6" gutterBottom>Global Pipeline Settings</Typography>
                                        <Divider sx={{ mb: 2 }}/>
                                        <Grid container spacing={2} alignItems="center"> {/* Nested container for this section's fields */}
                                            <Grid item xs={6} sm={3}>
                                                <Field as={TextField} name="imgSizeH" label="Img Height" type="number" fullWidth InputLabelProps={{ shrink: true }} />
                                            </Grid>
                                            <Grid item xs={6} sm={3}>
                                                <Field as={TextField} name="imgSizeW" label="Img Width" type="number" fullWidth InputLabelProps={{ shrink: true }} />
                                            </Grid>
                                            <Grid item xs={12} sm={3} sx={{ display: 'flex', alignItems: 'center', justifyContent: {xs: 'flex-start', sm: 'center'} }}>
                                                <FormControlLabel control={<Field as={Switch} type="checkbox" name="offlineAugmentation" />} label="Use Offline Augmented Data" />
                                            </Grid>
                                            <Grid item xs={12} sm={6}> {/* Adjust sm if 3 was too small overall */}
                                                <FormControl fullWidth error={touched.augmentationStrategyOverride && Boolean(errors.augmentationStrategyOverride)}>
                                                    <InputLabel id="augmentation-override-label">
                                                        Augmentation Override (Optional)
                                                    </InputLabel>
                                                    <Field
                                                        as={Select}
                                                        name="augmentationStrategyOverride"
                                                        labelId="augmentation-override-label"
                                                        label="Augmentation Override (Optional)"
                                                        sx={{ // Key: Style the Select component, specifically its display area
                                                            '& .MuiSelect-select': {
                                                                minWidth: '150px', // <<<< KEEP THIS OR ADJUST AS NEEDED
                                                                                   // This ensures the display box has a minimum width
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                                whiteSpace: 'nowrap', // Keep selected value on one line
                                                            },
                                                        }}
                                                        renderValue={(selectedValue) => {
                                                            if (!selectedValue || selectedValue === "") {
                                                                return <em style={{ opacity: 0.7 }}>(Pipeline Default)</em>;
                                                            }
                                                            const strategy = AVAILABLE_AUG_STRATEGIES.find(s => s.value === selectedValue);
                                                            return strategy ? strategy.label : selectedValue;
                                                        }}
                                                        MenuProps={{ // For the dropdown menu itself
                                                            PaperProps: {
                                                                style: {
                                                                    maxHeight: 280, // Max height of the dropdown list
                                                                    minWidth: 250,  // Ensure the dropdown LIST is wide enough for options
                                                                                    // This can be wider than the Select input itself
                                                                },
                                                            },
                                                        }}
                                                    >
                                                        <MenuItem value="">
                                                            <em style={{ opacity: 0.7 }}>(Pipeline Default)</em>
                                                        </MenuItem>
                                                        {AVAILABLE_AUG_STRATEGIES.map(aug => (
                                                            <MenuItem
                                                                key={aug.value}
                                                                value={aug.value}
                                                                sx={{ whiteSpace: 'normal' }} // Allow item text to wrap in the dropdown list
                                                            >
                                                                {aug.label}
                                                            </MenuItem>
                                                        ))}
                                                    </Field>
                                                    {touched.augmentationStrategyOverride && errors.augmentationStrategyOverride && (
                                                        <FormHelperText error>{errors.augmentationStrategyOverride}</FormHelperText>
                                                    )}
                                                </FormControl>
                                            </Grid>
                                            <Grid item xs={12} sm={6} md={3}>
                                                <Field
                                                    as={TextField}
                                                    name="test_split_ratio_if_flat"
                                                    label="Test Split Ratio (if flat)"
                                                    type="number"
                                                    fullWidth
                                                    size="small" /* If other fields are small */
                                                    InputLabelProps={{ shrink: true }}
                                                    InputProps={{inputProps: {step: "0.01", min:"0.01", max:"0.99"}}}
                                                    helperText="For FLAT datasets only"
                                                />
                                            </Grid>
                                            <Grid item xs={12} sm={6} md={3}>
                                                <Field
                                                    as={TextField}
                                                    name="random_seed"
                                                    label="Random Seed (Optional)"
                                                    type="number"
                                                    fullWidth
                                                    size="small"
                                                    InputLabelProps={{ shrink: true }}
                                                    helperText="Leave empty for default"
                                                />
                                            </Grid>
                                            <Grid item xs={12} sm={6} md={3} sx={{ display: 'flex', alignItems: 'center', justifyContent: {xs: 'flex-start', sm: 'center'} }}>
                                                <FormControlLabel
                                                    control={<Field as={Switch} type="checkbox" name="force_flat_for_fixed_cv" />}
                                                    label="Force Flat for Fixed CV"
                                                />
                                                <Tooltip title="If dataset has fixed train/test splits, this allows CV methods to use the entire dataset as one pool (use with caution).">
                                                    <IconButton size="small" sx={{p:0, ml:0.5}}><HelpOutlineIcon fontSize="inherit"/></IconButton>
                                                </Tooltip>
                                            </Grid>
                                        </Grid>
                                    </Grid>


                                    {/* Section 3: Methods Sequence */}
                                    <Grid item xs={12} sx={{ mt: 3 }}> {/* This Grid item spans full width for this section */}
                                        <Typography variant="h6" gutterBottom>Methods Sequence</Typography>
                                        <Divider sx={{ mb: 2 }}/>
                                        {/* The FieldArray for methodsSequence will go here, and it's already within its own full-width Grid item implicitly */}
                                        <FieldArray name="methodsSequence">
                                            {({ push, remove }) => (
                                                <Grid item xs={12}>
                                                    {values.methodsSequence.map((methodItem, idx) => ( // Use 'idx' here for map's index
                                                        <MethodStepCard
                                                            key={`method-${idx}-${methodItem.method_name}`}
                                                            method={methodItem}
                                                            index={idx} // Pass 'idx' as 'index' prop to MethodStepCard
                                                            values={values} errors={errors}
                                                            touched={touched} handleChange={handleChange}
                                                            setFieldValue={setFieldValue} remove={remove}
                                                            // Pass previous method name for conditional rendering of "use best params"
                                                            previousMethodName={idx > 0 ? values.methodsSequence[idx-1].method_name : null}
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
                                        {values.methodsSequence.length > 2 && (
                                            <Grid item xs={12}> {/* Ensure Alert is within a Grid item for proper layout */}
                                                <Alert severity="warning" sx={{ mt: 2 }}>
                                                    Using more than two methods in a custom sequence can be complex. Proceed with caution.
                                                </Alert>
                                            </Grid>
                                        )}
                                    </Grid>

                                    {/* Submit Button */}
                                    <Grid item xs={12} sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
                                        <Button
                                            type="submit"
                                            variant="contained"
                                            color="primary"
                                            size="large"
                                            disabled={isSubmitting || success}
                                            startIcon={isSubmitting ? <CircularProgress size={20} color="inherit" /> : null}
                                            sx={{ minWidth: 200 }} // Example: Set a minimum width
                                        >
                                            {isSubmitting ? 'Submitting...' : (success ? 'Submitted!' : 'Create and Run Experiment')}
                                        </Button>
                                    </Grid>
                                </Grid>
                            </Form>
                        )}
                    </Formik>
                </Paper>
            </Container>
        </>
    );
};

export default CreateExperimentPage;