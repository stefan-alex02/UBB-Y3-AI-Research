// src/components/Modals/NewPredictionModal.js
import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Button,
    CircularProgress,
    Alert,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    FormGroup,
    FormControlLabel,
    Checkbox,
    TextField,
    Box,
    Typography,
    FormHelperText
} from '@mui/material';
import experimentService from '../../services/experimentService'; // To fetch available models
import predictionService from '../../services/predictionService'; // To create prediction

const NewPredictionModal = ({ open, onClose, imageId, imageFormat, onPredictionCreated }) => {
    const [availableModels, setAvailableModels] = useState([]);
    const [selectedModelExperimentId, setSelectedModelExperimentId] = useState('');
    const [loadingModels, setLoadingModels] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [predictConfig, setPredictConfig] = useState({
        generateLime: false,
        limeNumFeatures: 5,
        limeNumSamples: 100,
        probPlotTopK: 5, // Default to top 5
    });

    useEffect(() => {
        if (open) {
            setError('');
            setLoadingModels(true);
            // Fetch experiments that have a model_relative_path (i.e., completed and saved a model)
            // And are not FAILED
            experimentService.getExperiments(
                { /* status: 'COMPLETED' - might be too restrictive, allow any non-failed with model path */ },
                { page: 0, size: 200, sortBy: 'startTime', sortDir: 'DESC' } // Fetch a good number of recent ones
            )
                .then(data => {
                    const models = data.content.filter(exp =>
                        exp.modelRelativePath && exp.modelRelativePath.trim() !== '' && exp.status !== 'FAILED'
                    );
                    setAvailableModels(models);
                    if (models.length > 0) {
                        // Optionally pre-select the first or most recent model
                        // setSelectedModelExperimentId(models[0].experimentRunId);
                    } else {
                        setSelectedModelExperimentId('');
                    }
                })
                .catch(err => {
                    console.error("Failed to load available models:", err);
                    setError('Failed to load available models: ' + (err.response?.data?.message || err.message));
                })
                .finally(() => setLoadingModels(false));
        } else {
            // Reset when modal closes
            setSelectedModelExperimentId('');
            setPredictConfig({
                generateLime: false, limeNumFeatures: 5, limeNumSamples: 100, probPlotTopK: 5
            });
        }
    }, [open]);

    const handleConfigChange = (event) => {
        const { name, value, checked, type } = event.target;
        setPredictConfig(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : (type === 'number' ? parseInt(value) || 0 : value)
        }));
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
                imageId: imageId, // imageId is passed as prop
                modelExperimentRunId: selectedModelExperimentId,
                generateLime: predictConfig.generateLime,
                limeNumFeatures: predictConfig.limeNumFeatures,
                limeNumSamples: predictConfig.limeNumSamples,
                probPlotTopK: predictConfig.probPlotTopK,
            });
            onPredictionCreated(); // Callback to refresh parent component
            onClose(); // Close modal
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to create prediction.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle>New Prediction for Image ID: {imageId}</DialogTitle>
            <DialogContent>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {loadingModels ? <Box sx={{display: 'flex', justifyContent:'center', my:2}}><CircularProgress /></Box> : (
                    <FormControl fullWidth margin="normal" disabled={availableModels.length === 0}>
                        <InputLabel id="model-select-label">Select Model (from Experiment)</InputLabel>
                        <Select
                            labelId="model-select-label"
                            value={selectedModelExperimentId}
                            label="Select Model (from Experiment)"
                            onChange={(e) => setSelectedModelExperimentId(e.target.value)}
                            required
                        >
                            {availableModels.length === 0 && <MenuItem value="" disabled><em>No trained models available</em></MenuItem>}
                            {availableModels.map(exp => (
                                <MenuItem key={exp.experimentRunId} value={exp.experimentRunId}>
                                    <Box sx={{display: 'flex', flexDirection: 'column', width: '100%'}}>
                                        <Typography variant="body1" component="span" sx={{fontWeight: 'medium'}}>
                                            {exp.name}
                                        </Typography>
                                        <Typography variant="caption" component="span" color="textSecondary">
                                            {exp.modelType} on {exp.datasetName} (Run: ...{exp.experimentRunId.slice(-8)})
                                        </Typography>
                                    </Box>
                                </MenuItem>
                            ))}
                        </Select>
                        {availableModels.length === 0 && !loadingModels && <FormHelperText>No suitable models found. Ensure experiments are completed and have saved models.</FormHelperText>}
                    </FormControl>
                )}

                <Typography variant="subtitle1" sx={{mt: 2, mb: 1}}>Prediction Options:</Typography>
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

export default NewPredictionModal;