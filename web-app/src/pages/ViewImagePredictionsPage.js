import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    Container,
    Typography,
    Paper,
    Box,
    CircularProgress,
    Alert,
    Button,
    Grid,
    Card,
    CardContent,
    CardMedia,
    Divider,
    Tabs,
    Tab,
    List,
    ListItem,
    ListItemButton,
    ListItemText,
    ListItemIcon,
    Chip
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import AssessmentIcon from '@mui/icons-material/Assessment'; // For predictions
import imageService from '../services/imageService';
import predictionService from '../services/predictionService';
import experimentService from '../services/experimentService'; // To list models for new prediction
import ArtifactViewer from '../components/ArtifactViewer/ArtifactViewer';
import LoadingSpinner from '../components/LoadingSpinner';
import useAuth from '../hooks/useAuth';
import { API_BASE_URL } from '../config'; // For constructing direct artifact URLs


// --- Modal for Creating New Prediction (Simplified) ---
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import TextField from '@mui/material/TextField';
import FolderIcon from "@mui/icons-material/Folder";
import ArticleIcon from "@mui/icons-material/Article";


// (getArtifactType helper function can be reused or put in a utils file)
const getArtifactType = (filename) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(extension)) return 'image';
    if (extension === 'json') return 'json';
    if (extension === 'log' || extension === 'txt') return 'log'; // Though unlikely for predictions
    if (extension === 'csv') return 'csv';
    return 'unknown';
};


const NewPredictionModal = ({ open, onClose, imageId, onPredictionCreated }) => {
    const [availableModels, setAvailableModels] = useState([]); // List of ExperimentDTOs
    const [selectedModelExperimentId, setSelectedModelExperimentId] = useState('');
    const [loadingModels, setLoadingModels] = useState(false);
    const [error, setError] = useState('');
    const [predictConfig, setPredictConfig] = useState({
        generateLime: false,
        limeNumFeatures: 5,
        limeNumSamples: 100,
        probPlotTopK: 5,
    });

    useEffect(() => {
        if (open) {
            setLoadingModels(true);
            // Fetch experiments that have a model_relative_path (i.e., completed and saved a model)
            experimentService.getExperiments({ status: 'COMPLETED' /* add other filters if needed */ }, { page: 0, size: 100, sortBy: 'startTime', sortDir: 'DESC' })
                .then(data => {
                    const models = data.content.filter(exp => exp.modelRelativePath && exp.modelRelativePath.trim() !== '');
                    setAvailableModels(models);
                    if (models.length > 0) {
                        setSelectedModelExperimentId(models[0].experimentRunId);
                    }
                })
                .catch(err => setError('Failed to load available models: ' + err.message))
                .finally(() => setLoadingModels(false));
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
        try {
            await predictionService.createPrediction({
                imageId: imageId,
                modelExperimentRunId: selectedModelExperimentId,
                ...predictConfig
            });
            onPredictionCreated();
            onClose();
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to create prediction.');
        }
    };

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle>Create New Prediction for Image ID: {imageId}</DialogTitle>
            <DialogContent>
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {loadingModels ? <CircularProgress /> : (
                    <FormControl fullWidth margin="normal" disabled={availableModels.length === 0}>
                        <InputLabel id="model-select-label">Select Model (from Experiment)</InputLabel>
                        <Select
                            labelId="model-select-label"
                            value={selectedModelExperimentId}
                            label="Select Model (from Experiment)"
                            onChange={(e) => setSelectedModelExperimentId(e.target.value)}
                        >
                            {availableModels.length === 0 && <MenuItem value="" disabled>No models available</MenuItem>}
                            {availableModels.map(exp => (
                                <MenuItem key={exp.experimentRunId} value={exp.experimentRunId}>
                                    {exp.name} ({exp.modelType} on {exp.datasetName}) - Run: ...{exp.experimentRunId.slice(-6)}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                )}
                <FormGroup sx={{mt: 2}}>
                    <FormControlLabel
                        control={<Checkbox checked={predictConfig.generateLime} onChange={handleConfigChange} name="generateLime" />}
                        label="Generate LIME Explanation"
                    />
                </FormGroup>
                {predictConfig.generateLime && (
                    <>
                        <TextField
                            margin="dense" label="LIME Features to Show" name="limeNumFeatures" type="number" fullWidth variant="standard"
                            value={predictConfig.limeNumFeatures} onChange={handleConfigChange}
                        />
                        <TextField
                            margin="dense" label="LIME Samples" name="limeNumSamples" type="number" fullWidth variant="standard"
                            value={predictConfig.limeNumSamples} onChange={handleConfigChange}
                        />
                    </>
                )}
                <TextField
                    margin="dense" label="Top K Probabilities to Plot" name="probPlotTopK" type="number" fullWidth variant="standard"
                    value={predictConfig.probPlotTopK} onChange={handleConfigChange} helperText="-1 for all classes"
                />
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Button onClick={handleSubmit} variant="contained" disabled={!selectedModelExperimentId || loadingModels}>Predict</Button>
            </DialogActions>
        </Dialog>
    );
};


const ViewImagePredictionsPage = () => {
    const { imageId } = useParams();
    const navigate = useNavigate();
    const { user } = useAuth();

    const [imageDetails, setImageDetails] = useState(null);
    const [predictions, setPredictions] = useState([]);
    const [selectedPrediction, setSelectedPrediction] = useState(null); // The full PredictionDTO
    const [artifacts, setArtifacts] = useState([]);
    const [selectedArtifactContent, setSelectedArtifactContent] = useState(null); // { name, type, content, url }
    const [currentArtifactPath, setCurrentArtifactPath] = useState(''); // For prediction artifacts sub-navigation

    const [isLoadingImage, setIsLoadingImage] = useState(true);
    const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
    const [isLoadingArtifacts, setIsLoadingArtifacts] = useState(false);
    const [error, setError] = useState(null);

    const [modalOpen, setModalOpen] = useState(false);

    const imageUrl = user && imageDetails ? `${API_BASE_URL}/python-proxy-images/${user.username}/${imageDetails.id}.${imageDetails.format}` : '';

    const fetchImageAndPredictions = useCallback(async () => {
        setIsLoadingImage(true);
        setIsLoadingPredictions(true);
        setError(null);
        try {
            if (user) { // Ensure user is available for username
                const imgData = await imageService.getImageByIdForUser(imageId, user.username);
                setImageDetails(imgData);
                const predData = await predictionService.getPredictionsForImage(imageId);
                setPredictions(predData);
                if (predData.length > 0) {
                    handlePredictionSelect(predData[0]); // Auto-select first prediction
                }
            }
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to load data.');
        } finally {
            setIsLoadingImage(false);
            setIsLoadingPredictions(false);
        }
    }, [imageId, user]);

    useEffect(() => {
        fetchImageAndPredictions();
    }, [fetchImageAndPredictions]);

    const fetchPredictionArtifacts = useCallback(async (prediction, subPath = '') => {
        if (!user || !prediction) return;
        setIsLoadingArtifacts(true);
        setSelectedArtifactContent(null);
        try {
            const data = await predictionService.listPredictionArtifacts(
                user.username,
                prediction.imageId,
                prediction.modelExperimentRunId,
                subPath
            );
            setArtifacts(data);
            setCurrentArtifactPath(subPath);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to list prediction artifacts.');
            setArtifacts([]);
        } finally {
            setIsLoadingArtifacts(false);
        }
    }, [user]);


    const handlePredictionSelect = (prediction) => {
        setSelectedPrediction(prediction);
        fetchPredictionArtifacts(prediction); // Fetch root artifacts for this prediction
    };

    const handleArtifactClick = async (artifactNode) => {
        if (!user || !selectedPrediction) return;
        if (artifactNode.type === 'folder') {
            fetchPredictionArtifacts(selectedPrediction, artifactNode.path);
        } else {
            setIsLoadingArtifacts(true);
            setSelectedArtifactContent({ name: artifactNode.name, type: 'loading' });
            try {
                const type = getArtifactType(artifactNode.name);
                const artifactBasePath = `${API_BASE_URL}/python-proxy-artifacts/predictions/${user.username}/${selectedPrediction.imageId}/${selectedPrediction.modelExperimentRunId}`;

                if (type === 'image') {
                    setSelectedArtifactContent({
                        name: artifactNode.name, type: type,
                        url: `${artifactBasePath}/${artifactNode.path}`, content: null
                    });
                } else {
                    const content = await predictionService.getPredictionArtifactContent(
                        user.username, selectedPrediction.imageId, selectedPrediction.modelExperimentRunId, artifactNode.path
                    );
                    setSelectedArtifactContent({ name: artifactNode.name, type: type, content: content, url: null });
                }
            } catch (err) {
                setError(`Failed to load artifact ${artifactNode.name}: ${err.message}`);
                setSelectedArtifactContent({ name: artifactNode.name, type: 'error', content: err.message });
            } finally {
                setIsLoadingArtifacts(false);
            }
        }
    };

    const handlePredictionCreated = () => {
        fetchImageAndPredictions(); // Refresh everything
    }


    if (isLoadingImage) return <LoadingSpinner />;
    if (error && !imageDetails) return <Container><Alert severity="error" sx={{mt:2}}>{error}</Alert></Container>;

    return (
        <Container maxWidth="xl" sx={{ mt: 2 }}>
            <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/images')} sx={{ mb: 2 }}>
                Back to Images
            </Button>

            <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                    {imageDetails && (
                        <Paper elevation={3} sx={{ p: 2, mb: 2, position: 'sticky', top: '80px' /* Adjust based on TopBar height */ }}>
                            <Typography variant="h5" gutterBottom>Image: {imageDetails.id}.{imageDetails.format}</Typography>
                            <CardMedia component="img" image={imageUrl} alt={`Image ${imageDetails.id}`} sx={{ borderRadius: 1, maxHeight: 300, objectFit: 'contain', mb: 2 }} />
                            <Typography variant="body2">Uploaded: {new Date(imageDetails.uploadedAt).toLocaleString()}</Typography>
                            <Button
                                variant="contained"
                                startIcon={<AddCircleOutlineIcon />}
                                onClick={() => setModalOpen(true)}
                                fullWidth
                                sx={{ mt: 2 }}
                            >
                                New Prediction
                            </Button>
                        </Paper>
                    )}

                    <Paper elevation={1} sx={{p:1, mt:2}}>
                        <Typography variant="h6" sx={{p:1}}>Predictions History</Typography>
                        {isLoadingPredictions ? <CircularProgress sx={{m:2}} /> : (
                            <List dense>
                                {predictions.map(pred => (
                                    <ListItemButton key={pred.id} selected={selectedPrediction?.id === pred.id} onClick={() => handlePredictionSelect(pred)}>
                                        <ListItemIcon sx={{minWidth: '36px'}}><AssessmentIcon fontSize="small" color={selectedPrediction?.id === pred.id ? "primary" : "action"}/></ListItemIcon>
                                        <ListItemText
                                            primary={`${pred.predictedClass} (${(pred.confidence * 100).toFixed(1)}%)`}
                                            secondary={`Model: ...${pred.modelExperimentRunId.slice(-6)} on ${new Date(pred.predictionTimestamp).toLocaleDateString()}`}
                                            primaryTypographyProps={{variant: 'body2', fontWeight: selectedPrediction?.id === pred.id ? 'bold' : 'normal'}}
                                        />
                                    </ListItemButton>
                                ))}
                                {predictions.length === 0 && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No predictions yet for this image.</Typography>}
                            </List>
                        )}
                    </Paper>
                </Grid>

                <Grid item xs={12} md={8}>
                    {selectedPrediction ? (
                        <Paper elevation={2} sx={{ p: 2 }}>
                            <Typography variant="h5" gutterBottom>
                                Prediction Details (Model: ...{selectedPrediction.modelExperimentRunId.slice(-6)})
                            </Typography>
                            <Typography variant="h6">Predicted Class: <Chip label={selectedPrediction.predictedClass} color="primary" /></Typography>
                            <Typography variant="subtitle1">Confidence: {(selectedPrediction.confidence * 100).toFixed(2)}%</Typography>
                            <Typography variant="body2" color="text.secondary">Timestamp: {new Date(selectedPrediction.predictionTimestamp).toLocaleString()}</Typography>
                            <Divider sx={{ my: 2 }} />
                            <Typography variant="h6">Artifacts</Typography>
                            {/* TODO: Breadcrumbs for prediction artifacts if they have subfolders, similar to ViewExperimentPage */}
                            {isLoadingArtifacts && artifacts.length === 0 ? <LoadingSpinner /> : (
                                <List dense>
                                    {artifacts.sort((a,b) => a.type.localeCompare(b.type) || a.name.localeCompare(b.name)).map((node) => (
                                        <ListItem key={node.path} disablePadding>
                                            <ListItemButton onClick={() => handleArtifactClick(node)} selected={selectedArtifactContent?.name === node.name}>
                                                <ListItemIcon sx={{minWidth: '32px'}}>{node.type === 'folder' ? <FolderIcon fontSize="small" /> : <ArticleIcon fontSize="small" />}</ListItemIcon>
                                                <ListItemText primary={node.name} primaryTypographyProps={{ variant: 'body2', noWrap: true }} />
                                            </ListItemButton>
                                        </ListItem>
                                    ))}
                                    {artifacts.length === 0 && !isLoadingArtifacts && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No artifacts for this prediction.</Typography>}
                                </List>
                            )}
                            <Box sx={{ mt: 2, border: '1px dashed grey', p: selectedArtifactContent ? 0 : 2, borderRadius: 1, minHeight: 200 }}>
                                {isLoadingArtifacts && selectedArtifactContent?.type === 'loading' && <LoadingSpinner />}
                                {selectedArtifactContent && selectedArtifactContent.type !== 'loading' && (
                                    <ArtifactViewer
                                        artifactName={selectedArtifactContent.name}
                                        artifactType={selectedArtifactContent.type}
                                        artifactContent={selectedArtifactContent.content}
                                        artifactUrl={selectedArtifactContent.url}
                                    />
                                )}
                                {!selectedArtifactContent && !isLoadingArtifacts && <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: '10%' }}>Select an artifact.</Typography>}
                            </Box>
                        </Paper>
                    ) : (
                        <Paper elevation={2} sx={{ p: 3, textAlign: 'center' }}>
                            <Typography variant="h6" color="text.secondary">
                                Select a prediction from the list to view details or create a new one.
                            </Typography>
                        </Paper>
                    )}
                </Grid>
            </Grid>

            {imageDetails && (
                <NewPredictionModal
                    open={modalOpen}
                    onClose={() => setModalOpen(false)}
                    imageId={imageDetails.id}
                    onPredictionCreated={handlePredictionCreated}
                />
            )}
        </Container>
    );
};

export default ViewImagePredictionsPage;