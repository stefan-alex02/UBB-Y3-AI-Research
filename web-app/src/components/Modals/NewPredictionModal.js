// src/components/Modals/NewPredictionModal.js
import React, {useCallback, useEffect, useState} from 'react';
import {
    Alert,
    Box,
    Button,
    Checkbox,
    CircularProgress,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Divider,
    FormControl,
    FormControlLabel,
    FormGroup,
    Grid,
    InputLabel,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    MenuItem,
    Pagination,
    Paper,
    Select,
    TextField,
    Typography
} from '@mui/material';
import experimentService from '../../services/experimentService'; // To fetch available models
import predictionService from '../../services/predictionService';
import * as PropTypes from "prop-types";
import {format as formatDateFns} from "date-fns";
import {DATASET_NAMES, MODEL_TYPES} from '../../pages/experimentConfig';
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import DatasetIcon from "@mui/icons-material/Dataset";
import PersonIcon from "@mui/icons-material/Person";
import EventIcon from "@mui/icons-material/Event";


const formatDateSafe = (timestampSeconds) => {
    if (timestampSeconds === null || timestampSeconds === undefined) {
        return 'N/A';
    }
    try {
        const date = new Date(Number(timestampSeconds) * 1000); // Convert seconds to ms
        if (isNaN(date.getTime())) { // Check if date is valid
            return 'Invalid Date';
        }
        // Use date-fns format for consistency and better formatting options
        return formatDateFns(date, 'PPpp'); // Example: 'Jul 2, 2021, 5:07:59 PM'
        // Or a simpler custom format:
        // return date.toLocaleString([], { year: 'numeric', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        console.error("Error formatting date:", timestampSeconds, e);
        return 'Date Error';
    }
};

const ModelSelectItem = ({ experiment, selected, onClick }) => (
    <ListItemButton
        selected={selected}
        onClick={onClick}
        sx={{
            mb: 1,
            borderRadius: 1,
            border: selected ? '2px solid' : '1px solid',
            borderColor: selected ? 'primary.main' : 'divider',
            p: 1.5
        }}
    >
        <ListItemIcon sx={{minWidth: 36, mr: 1.5}}><ModelTrainingIcon color={selected ? "primary" : "action"} fontSize="medium"/></ListItemIcon>
        <ListItemText
            primary={
                <Typography variant="body1" component="div" sx={{ fontWeight: selected ? 500 : 400, color: selected ? "primary.main" : "text.primary", mb: 0.5 }}>
                    {experiment.name || "Unnamed Experiment"}
                </Typography>
            }
            secondary={
                <Box>
                    <Typography component="div" variant="caption" color="text.secondary" sx={{display: 'flex', alignItems: 'center', mb: 0.25}}>
                        <ModelTrainingIcon fontSize="inherit" sx={{mr:0.5, opacity:0.7}}/> Model: {experiment.model_type || 'N/A'}
                    </Typography>
                    <Typography component="div" variant="caption" color="text.secondary" sx={{display: 'flex', alignItems: 'center', mb: 0.25}}>
                        <DatasetIcon fontSize="inherit" sx={{mr:0.5, opacity:0.7}}/> Dataset: {experiment.dataset_name || 'N/A'}
                    </Typography>
                    <Typography component="div" variant="caption" color="text.secondary" sx={{display: 'flex', alignItems: 'center', mb: 0.25}}>
                        <PersonIcon fontSize="inherit" sx={{mr:0.5, opacity:0.7}}/> By: {experiment.user_name || 'N/A'}
                    </Typography>
                    <Typography component="div" variant="caption" color="text.secondary" sx={{display: 'flex', alignItems: 'center'}}>
                        <EventIcon fontSize="inherit" sx={{mr:0.5, opacity:0.7}}/> Completed: {formatDateSafe(experiment.end_time)}
                    </Typography>
                    <Typography component="div" variant="caption" color="text.secondary" sx={{mt:0.25, fontFamily: 'monospace', fontSize:'0.7rem'}}>
                        ID: ...{experiment.experiment_run_id ? experiment.experiment_run_id.slice(-12) : 'N/A'}
                    </Typography>
                </Box>
            }
        />
    </ListItemButton>
);

ModelSelectItem.propTypes = {
    experiment: PropTypes.shape({ // More specific PropTypes
        name: PropTypes.string,
        experiment_run_id: PropTypes.string.isRequired,
        model_type: PropTypes.string,
        dataset_name: PropTypes.string,
        user_name: PropTypes.string,
        end_time: PropTypes.number, // Assuming it's a number (timestamp)
    }).isRequired,
    selected: PropTypes.bool,
    onClick: PropTypes.func
};

const NewPredictionModal = ({ open, onClose, imageIds, onPredictionCreated }) => {
    const [availableModels, setAvailableModels] = useState([]);
    const [selectedModelExperimentId, setSelectedModelExperimentId] = useState('');

    const [modelFilters, setModelFilters] = useState({
        nameContains: '', // New filter
        modelType: '',
        datasetName: ''
    });
    const [modelPagination, setModelPagination] = useState({ page: 0, size: 4, totalPages: 0 });

    const [loadingModels, setLoadingModels] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [predictConfig, setPredictConfig] = useState({
        generateLime: false,
        limeNumFeatures: 5,
        limeNumSamples: 100,
        probPlotTopK: 5,
    });

    const fetchAvailableModels = useCallback(async () => {
        if (!open) return; // Ensure we only fetch if the modal is intended to be open

        setLoadingModels(true);
        setError('');
        try {
            const filters = {
                status: 'COMPLETED',
                hasModelSaved: true,
                nameContains: modelFilters.nameContains || undefined,
                modelType: modelFilters.modelType || undefined,
                datasetName: modelFilters.datasetName || undefined,
            };
            const pageable = { page: modelPagination.page, size: modelPagination.size, sortBy: 'endTime', sortDir: 'DESC' };

            const data = await experimentService.getExperiments(filters, pageable);
            const fetchedModels = data.content || [];
            setAvailableModels(fetchedModels);
            setModelPagination(prev => ({ ...prev, totalPages: data.totalPages }));

            // If the currently selected model is no longer in the new list, clear the selection.
            // This check should happen *after* setting availableModels.
            if (selectedModelExperimentId && !fetchedModels.some(m => m.experiment_run_id === selectedModelExperimentId)) {
                setSelectedModelExperimentId('');
            }

        } catch (err) {
            setError('Failed to load available models: ' + (err.response?.data?.message || err.message));
            setAvailableModels([]);
        } finally {
            setLoadingModels(false);
        }
        // `selectedModelExperimentId` is removed from here to break the loop.
        // The logic to clear it is now inside the function, and this function will be
        // re-called if filters/pagination change, at which point the check is still valid.
    }, [open, modelFilters, modelPagination.page, modelPagination.size]);


    useEffect(() => {
        // This effect handles fetching when modal opens or filters/pagination change
        if (open) {
            fetchAvailableModels();
        }
    }, [open, fetchAvailableModels]); // fetchAvailableModels is now more stable

    useEffect(() => {
        // This effect handles resetting state when the modal is closed
        if (!open) {
            setSelectedModelExperimentId('');
            setModelFilters({ nameContains: '', modelType: '', datasetName: '' });
            setModelPagination(prev => ({ ...prev, page: 0, totalPages: 0 }));
            setAvailableModels([]);
            setError('');
            setPredictConfig({ generateLime: false, limeNumFeatures: 5, limeNumSamples: 100, probPlotTopK: 5 });
        }
    }, [open]); // Only depends on 'open'

    const handleConfigChange = (event) => {
        const { name, value, checked, type } = event.target;
        setPredictConfig(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : (type === 'number' ? parseInt(value) || 0 : value)
        }));
    };
    const handleModelFilterChange = (event) => {
        const { name, value } = event.target;
        setModelFilters(prev => ({ ...prev, [name]: value }));
        setModelPagination(prev => ({ ...prev, page: 0 })); // Reset page on filter change
    };
    const handleModelPageChange = (event, value) => {
        setModelPagination(prev => ({ ...prev, page: value - 1 }));
    };

    const handleSubmit = async () => {
        if (!selectedModelExperimentId) {
            setError("Please select a model.");
            return;
        }
        setError('');
        setSubmitting(true);
        try {
            // The PredictionCreateRequest on Java side expects imageId and modelExperimentRunId
            // and the Python config params.
            // The PythonPredictionRequestDTO built in Java service will construct the full details.
            await predictionService.createPrediction({
                image_ids: imageIds, // imageId is passed as prop
                model_experiment_run_id: selectedModelExperimentId,
                generate_lime: predictConfig.generateLime,
                lime_num_features: predictConfig.limeNumFeatures,
                lime_num_samples: predictConfig.limeNumSamples,
                prob_plot_top_k: predictConfig.probPlotTopK,
            });
            if (onPredictionCreated) onPredictionCreated();
            onClose(); // Close modal
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to create prediction.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle>
                New Prediction for {imageIds.length > 1 ? `${imageIds.length} Images` : `Image ID: ${imageIds[0]}`}
            </DialogTitle>
            <DialogContent dividers>
                {error && <Alert severity="error" sx={{ mb: 2 }} onClose={()=>setError('')}>{error}</Alert>}

                <Typography variant="h6" gutterBottom>Select Model</Typography>
                <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} sm={4}>
                            <TextField fullWidth label="Filter by Experiment Name" name="nameContains" value={modelFilters.nameContains} onChange={handleModelFilterChange} variant="outlined" size="small"/>
                        </Grid>
                        <Grid item xs={12} sm={4}>
                            <FormControl fullWidth size="small">
                                <InputLabel>Filter by Model Type</InputLabel>
                                <Select name="modelType" value={modelFilters.modelType} label="Filter by Model Type" onChange={handleModelFilterChange}>
                                    <MenuItem value=""><em>Any Type</em></MenuItem>
                                    {MODEL_TYPES.map(mt => <MenuItem key={mt.value} value={mt.value}>{mt.label}</MenuItem>)}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                            <FormControl fullWidth size="small">
                                <InputLabel>Filter by Dataset</InputLabel>
                                <Select name="datasetName" value={modelFilters.datasetName} label="Filter by Dataset" onChange={handleModelFilterChange}>
                                    <MenuItem value=""><em>Any Dataset</em></MenuItem>
                                    {DATASET_NAMES.map(dn => <MenuItem key={dn} value={dn}>{dn}</MenuItem>)}
                                </Select>
                            </FormControl>
                        </Grid>
                    </Grid>
                </Paper>

                {loadingModels ? <Box sx={{display:'flex', justifyContent:'center',my:3}}><CircularProgress /></Box> : (
                    <>
                        {availableModels.length === 0 ? (
                            <Typography sx={{textAlign:'center', my:3, color:'text.secondary'}}>No models match your criteria.</Typography>
                        ) : (
                            <Grid container spacing={2} sx={{ maxHeight: 350, overflow: 'auto', mb:1, pr:1 /* padding for scrollbar */ }}>
                                {availableModels.map(exp => (
                                    <Grid item xs={12} sm={6} key={exp.experiment_run_id}> {/* 2 cards per row on sm+ */}
                                        <ModelSelectItem
                                            experiment={exp}
                                            selected={selectedModelExperimentId === exp.experiment_run_id}
                                            onClick={() => setSelectedModelExperimentId(exp.experiment_run_id)}
                                        />
                                    </Grid>
                                ))}
                            </Grid>
                        )}
                        {availableModels.length > 0 && modelPagination.totalPages > 1 && (
                            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                                <Pagination count={modelPagination.totalPages} page={modelPagination.page + 1} onChange={handleModelPageChange} size="small"/>
                            </Box>
                        )}
                    </>
                )}

                <Divider sx={{my:2}} />
                <Typography variant="h6" gutterBottom sx={{mt:1}}>Prediction Options</Typography>
                <FormGroup>
                    <FormControlLabel
                        control={<Checkbox checked={predictConfig.generateLime} onChange={handleConfigChange} name="generateLime" />}
                        label="Generate LIME Explanation"
                    />
                </FormGroup>
                {predictConfig.generateLime && (
                    <Box sx={{pl:2}}>
                        <TextField
                            margin="dense" label="LIME Features to Show" name="limeNumFeatures" type="number" fullWidth variant="standard"
                            value={predictConfig.limeNumFeatures} onChange={handleConfigChange} InputLabelProps={{ shrink: true }}
                            InputProps={{ inputProps: { min: 1, max: 20 } }}
                        />
                        <TextField
                            margin="dense" label="LIME Samples" name="limeNumSamples" type="number" fullWidth variant="standard"
                            value={predictConfig.limeNumSamples} onChange={handleConfigChange} InputLabelProps={{ shrink: true }}
                            InputProps={{ inputProps: { min: 50, max: 2000, step: 50 } }}
                        />
                    </Box>
                )}
                <TextField
                    margin="normal" label="Top K Probabilities to Plot" name="probPlotTopK" type="number" fullWidth variant="standard"
                    value={predictConfig.probPlotTopK} onChange={handleConfigChange} helperText="-1 for all classes, 0 to disable plot"
                    InputLabelProps={{ shrink: true }}
                    InputProps={{ inputProps: { min: -1 } }}
                />
                {submitting && !error && (
                    <Alert severity="info" sx={{mt:2}}>
                        Prediction submitted! You can close this window.
                        Refresh the predictions list on the main page to see the result when ready.
                    </Alert>
                )}
            </DialogContent>
            <DialogActions sx={{p: '16px 24px'}}>
                <Button onClick={onClose} color="inherit">Cancel</Button>
                <Button
                    onClick={handleSubmit}
                    variant="contained"
                    disabled={!selectedModelExperimentId || loadingModels || submitting}
                >
                    {submitting ? <CircularProgress size={20} color="inherit"/> : "Run Prediction"}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

NewPredictionModal.propTypes = {
    open: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    imageIds: PropTypes.arrayOf(PropTypes.oneOfType([PropTypes.string, PropTypes.number])).isRequired,
    onPredictionCreated: PropTypes.func.isRequired,
};

export default NewPredictionModal;