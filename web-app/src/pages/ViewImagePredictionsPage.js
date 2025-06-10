import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {useNavigate, useParams} from 'react-router-dom';
import {
    Alert,
    Box,
    Button,
    CardMedia,
    Chip,
    CircularProgress,
    Container,
    Grid,
    IconButton,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Paper,
    Skeleton,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TableSortLabel,
    Tabs,
    Typography
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import AssessmentIcon from '@mui/icons-material/Assessment';
import BrokenImageIcon from '@mui/icons-material/BrokenImage';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import BarChartIcon from '@mui/icons-material/BarChart'; // For Probability Plot
import imageService from '../services/imageService';
import predictionService from '../services/predictionService';
import PlotViewer from '../components/ArtifactViewer/PlotViewer'; // For direct use with plots
import LoadingSpinner from '../components/LoadingSpinner';
import ImageFullscreenModal from '../components/ImageFullscreenModal';
import NewPredictionModal from '../components/Modals/NewPredictionModal';
import InsightsIcon from '@mui/icons-material/Insights';
import {API_BASE_URL} from "../config";
import {getComparator, stableSort} from "../utils/tableUtils";
import {formatDateSafe} from "../utils/dateUtils";
import useAuth from "../hooks/useAuth"; // Assuming modal is moved

// (getArtifactType helper function - ensure it's defined or imported from a utils file)
const getArtifactType = (filename) => {
    if (!filename) return 'unknown';
    const extension = filename.split('.').pop()?.toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(extension)) return 'image';
    if (extension === 'json') return 'json';
    if (extension === 'log' || extension === 'txt') return 'log';
    if (extension === 'csv') return 'csv';
    return 'unknown';
};

const ViewImagePredictionsPage = () => {
    const { imageId: routeImageId } = useParams();
    const navigate = useNavigate();
    const { user } = useAuth();

    const [imageDetails, setImageDetails] = useState(null);
    const [predictions, setPredictions] = useState([]);
    const [selectedPrediction, setSelectedPrediction] = useState(null);
    const [selectedPredictionDetailsJson, setSelectedPredictionDetailsJson] = useState(null);
    const [predictionArtifacts, setPredictionArtifacts] = useState([]);
    const [selectedPlotToView, setSelectedPlotToView] = useState(null); // Not used in the latest version, selectedArtifactToView covers plots

    // State for the main uploaded image display
    const [mainImageBlob, setMainImageBlob] = useState(null);
    const [isLoadingMainImage, setIsLoadingMainImage] = useState(true); // <<<< ENSURE THIS IS DECLARED
    const [mainImageError, setMainImageError] = useState(false);

    // State for general page loading and errors
    const [isLoadingPageData, setIsLoadingPageData] = useState(true); // Overall page data (image metadata + prediction list)
    const [isLoadingPredictionsList, setIsLoadingPredictionsList] = useState(false); // Specifically for the predictions list part if fetched separately
    const [isLoadingSelectedPredictionContent, setIsLoadingSelectedPredictionContent] = useState(false);
    const [pageError, setPageError] = useState(null);

    const [activeContentTab, setActiveContentTab] = useState(0);

    const [newPredictionModalOpen, setNewPredictionModalOpen] = useState(false);
    const [fullscreenModal, setFullscreenModal] = useState({ open: false, src: null, title: '', type: 'url' });

    const fetchPageInitialData = useCallback(async () => {
        if (!user || !routeImageId) return;
        setIsLoadingPageData(true);
        setIsLoadingMainImage(true);
        setMainImageError(false);
        setPageError(null);
        // tempMainImageBlob is not strictly needed if mainImageDisplayUrl's useEffect handles cleanup
        // let tempMainImageBlob = null;

        try {
            const imgData = await imageService.getImageByIdForUser(Number(routeImageId), user.username);
            setImageDetails(imgData);

            if (imgData && imgData.id) {
                try {
                    const blob = await imageService.getImageContentBlob(imgData.id);
                    setMainImageBlob(blob);
                } catch (contentError) {
                    console.error("Failed to load main image content:", contentError);
                    setMainImageError(true);
                }
            } else {
                setMainImageError(true);
            }
            setIsLoadingMainImage(false);

            setIsLoadingPredictionsList(true);
            const predData = await predictionService.getPredictionsForImage(Number(routeImageId));
            setPredictions(predData);
            // Do not auto-select, let user click
            setSelectedPrediction(null);
            setSelectedPredictionDetailsJson(null);
            setPredictionArtifacts([]);
            setIsLoadingPredictionsList(false);

        } catch (err) {
            setPageError(err.response?.data?.message || err.message || "Failed to load page data.");
            setMainImageError(true);
            setIsLoadingMainImage(false);
            setIsLoadingPredictionsList(false);
        } finally {
            setIsLoadingPageData(false);
        }
    }, [routeImageId, user]); // Removed user.username, user object itself is enough

    useEffect(() => {
        fetchPageInitialData();
    }, [fetchPageInitialData]);


    // Fetches content for a selected prediction (prediction_details.json and lists its plots)
    const fetchSelectedPredictionData = useCallback(async (prediction) => {
        if (!user || !prediction || !imageDetails) return;
        setIsLoadingSelectedPredictionContent(true);
        setSelectedPredictionDetailsJson(null);
        setPredictionArtifacts([]); // Clear old artifacts
        setPageError(null);

        try {
            // 1. Fetch prediction_details.json
            const jsonContentStr = await predictionService.getPredictionArtifactContent(
                user.username, String(imageDetails.id), prediction.modelExperimentRunId, "prediction_details.json"
            );
            setSelectedPredictionDetailsJson(JSON.parse(jsonContentStr));

            // 2. List artifacts in the "plots" subfolder for this prediction
            const plotArtifacts = await predictionService.listPredictionArtifacts(
                user.username, String(imageDetails.id), prediction.modelExperimentRunId, "plots"
            );
            setPredictionArtifacts(plotArtifacts.filter(art => getArtifactType(art.name) === 'image')); // Only image plots

        } catch (err) {
            setPageError(err.response?.data?.message || err.message || `Failed to load details for prediction made with model ${prediction.modelExperimentRunId}.`);
            setSelectedPredictionDetailsJson({ error: "Could not load details." });
        } finally {
            setIsLoadingSelectedPredictionContent(false);
        }
    }, [user, imageDetails]);

    const handlePredictionSelect = (prediction) => {
        setSelectedPrediction(prediction);
        setActiveContentTab(0); // Default to probabilities tab
        if (prediction) {
            fetchSelectedPredictionData(prediction);
        } else { // Deselected or no prediction
            setSelectedPredictionDetailsJson(null);
            setPredictionArtifacts([]);
        }
    };

    // Create object URL for main image display only when needed
    const mainImageDisplayUrl = useMemo(() => {
        if (mainImageBlob instanceof Blob) {
            return URL.createObjectURL(mainImageBlob);
        }
        return null;
    }, [mainImageBlob]);

    const openFullscreen = (src, title, type = 'url') => {
        setFullscreenModal({ open: true, src, title, type });
    };

    // CSV Sorting state for probabilities table
    const [probTableOrder, setProbTableOrder] = useState('desc');
    const [probTableOrderBy, setProbTableOrderBy] = useState('probability'); // Default sort by probability

    const handleProbTableSortRequest = (property) => {
        const isAsc = probTableOrderBy === property && probTableOrder === 'asc';
        setProbTableOrder(isAsc ? 'desc' : 'asc');
        setProbTableOrderBy(property);
    };

    const sortedProbabilities = useMemo(() => {
        if (!selectedPredictionDetailsJson || !selectedPredictionDetailsJson.top_k_predictions_for_plot) {
            return [];
        }
        // top_k_predictions_for_plot is [ [className, prob], ... ]
        // Convert to object array for easier sorting with stableSort
        const data = selectedPredictionDetailsJson.top_k_predictions_for_plot.map(p => ({ className: p[0], probability: p[1] }));
        return stableSort(data, getComparator(probTableOrder, probTableOrderBy));
    }, [selectedPredictionDetailsJson, probTableOrder, probTableOrderBy]);

    const getPlotUrl = (plotName) => {
        if (!user || !imageDetails || !selectedPrediction) return '';
        // This needs to call the Java proxy that gets content from Python
        return `${API_BASE_URL}/api/predictions/${imageDetails.id}/model/${selectedPrediction.modelExperimentRunId}/artifacts/content/plots/${plotName}`;
    };


    // --- Main Render ---
    if (isLoadingPageData && !imageDetails) return <Container sx={{mt:2, textAlign: 'center'}}><CircularProgress size={50} sx={{mt:5}}/><Typography>Loading image details...</Typography></Container>;
    if (pageError && !imageDetails) return <Container sx={{mt:2}}><Alert severity="error" onClose={() => setPageError(null)}>{pageError}</Alert></Container>;
    if (!imageDetails) return <Container sx={{mt:2}}><Alert severity="info">Image details not available or ID not found.</Alert></Container>;


    return (
        <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
            <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/images')} sx={{ mb: 2 }}>Back to Images</Button>

            <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                    <Paper elevation={3} sx={{ p: 2, mb: 2, position: 'sticky', top: '80px' }}>
                        <Typography variant="h5" gutterBottom>Image: {imageDetails.id}.{imageDetails.format}</Typography>
                        <Box sx={{ textAlign: 'center', mb: 1, position: 'relative' }}>
                            {isLoadingMainImage ? ( // Use isLoadingMainImage here
                                <Skeleton variant="rectangular" width="100%" height={250} animation="wave" sx={{ borderRadius: 1 }} />
                            ) : mainImageError || !mainImageDisplayUrl ? ( // Check for error OR if URL couldn't be created
                                <Box sx={{ height: 250, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: 'grey.200', borderRadius: 1 }}>
                                    <BrokenImageIcon color="action" sx={{ fontSize: 60 }} />
                                    <Typography color="textSecondary">Image Preview Unavailable</Typography>
                                </Box>
                            ) : (
                                <>
                                    <CardMedia component="img" image={mainImageDisplayUrl} alt={`Image ${imageDetails.id}`} sx={{ maxHeight: 300, objectFit: 'contain', borderRadius: 1 }} />
                                    <IconButton onClick={() => mainImageBlob && openFullscreen(mainImageBlob, `Image ${imageDetails.id}`, 'blob')} sx={{position:'absolute', top: 8, right: 8, bgcolor: 'rgba(0,0,0,0.3)', '&:hover':{bgcolor:'rgba(0,0,0,0.5)'}}} size="small" disabled={!mainImageBlob || mainImageError}>
                                        <ZoomInIcon sx={{color: 'white'}}/>
                                    </IconButton>
                                </>
                            )}
                        </Box>
                        <Typography variant="body2" color="text.secondary">Uploaded: {formatDateSafe(imageDetails.uploadedAt)}</Typography>
                        <Button fullWidth variant="contained" startIcon={<AddCircleOutlineIcon />} onClick={() => setNewPredictionModalOpen(true)} sx={{ mt: 2 }}>New Prediction</Button>
                    </Paper>

                    <Paper elevation={1} sx={{p:1, mt:2}}>
                        <Typography variant="h6" sx={{p:1}}>Predictions History</Typography>
                        {isLoadingPredictionsList && predictions.length === 0 ? <CircularProgress sx={{m:2}} /> : ( // Use isLoadingPredictionsList
                            <List dense sx={{maxHeight: 'calc(100vh - 550px)', overflowY: 'auto'}}>
                                {predictions.map(pred => (
                                    <ListItemButton key={pred.id} selected={selectedPrediction?.id === pred.id} onClick={() => handlePredictionSelect(pred)}>
                                        <ListItemIcon sx={{minWidth: '36px'}}><AssessmentIcon fontSize="small" color={selectedPrediction?.id === pred.id ? "primary" : "action"}/></ListItemIcon>
                                        <ListItemText
                                            primary={`${pred.predictedClass} (${(pred.confidence * 100).toFixed(1)}%)`}
                                            secondary={`Model: ${pred.modelExperimentName || (pred.modelExperimentRunId ? `...${pred.modelExperimentRunId.slice(-6)}` : 'Unknown')} | ${formatDateSafe(pred.predictionTimestamp)}`}
                                            primaryTypographyProps={{variant: 'body2', fontWeight: selectedPrediction?.id === pred.id ? 'bold' : 'normal'}}
                                        />
                                    </ListItemButton>
                                ))}
                                {predictions.length === 0 && !isLoadingPredictionsList && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No predictions recorded for this image.</Typography>}
                            </List>
                        )}
                    </Paper>
                </Grid>

                {/* Right Column: Selected Prediction Details & Artifacts */}
                <Grid item xs={12} md={8}>
                    {!selectedPrediction && !isLoadingSelectedPredictionContent && (
                        <Paper elevation={2} sx={{ p: 3, textAlign: 'center', minHeight: 400, display:'flex', alignItems:'center', justifyContent:'center' }}>
                            <Typography variant="h6" color="text.secondary">Select a prediction from the list to view details.</Typography>
                        </Paper>
                    )}
                    {isLoadingSelectedPredictionContent && <Box sx={{display:'flex', justifyContent:'center', alignItems:'center', height:400}}><CircularProgress /></Box>}

                    {selectedPrediction && selectedPredictionDetailsJson && !isLoadingSelectedPredictionContent && (
                        <Paper elevation={2} sx={{ p: 2 }}>
                            <Box sx={{mb:2}}>
                                <Typography variant="h5" gutterBottom>Details for Prediction with Model: {selectedPrediction.modelExperimentName || `...${selectedPrediction.modelExperimentRunId.slice(-6)}`}</Typography>
                                <Typography variant="h6">Main Prediction: <Chip label={`${selectedPredictionDetailsJson.predicted_class_name} (${(selectedPredictionDetailsJson.confidence * 100).toFixed(1)}%)`} color="primary" /></Typography>
                                <Typography variant="caption" display="block">Image Source Path: {selectedPredictionDetailsJson.image_user_source_path}</Typography>
                            </Box>

                            <Tabs value={activeContentTab} onChange={(e, newValue) => setActiveContentTab(newValue)} variant="fullWidth">
                                <Tab label="Class Probabilities & Plot" icon={<BarChartIcon/>} iconPosition="start" />
                                <Tab label="LIME Explanation" icon={<InsightsIcon/>} iconPosition="start" disabled={!selectedPredictionDetailsJson.lime_explanation || selectedPredictionDetailsJson.lime_explanation.error} />
                            </Tabs>

                            {/* Probabilities Tab Panel */}
                            <Box role="tabpanel" hidden={activeContentTab !== 0} sx={{pt:2}}>
                                {activeContentTab === 0 && (
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={5}> {/* Probabilities Table */}
                                            <Typography variant="subtitle1" gutterBottom>Top Class Probabilities</Typography>
                                            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 350 }}>
                                                <Table stickyHeader size="small">
                                                    <TableHead><TableRow>
                                                        <TableCell><TableSortLabel active={probTableOrderBy==='className'} direction={probTableOrderBy==='className' ? probTableOrder : 'asc'} onClick={()=>handleProbTableSortRequest('className')}>Class</TableSortLabel></TableCell>
                                                        <TableCell align="right"><TableSortLabel active={probTableOrderBy==='probability'} direction={probTableOrderBy==='probability' ? probTableOrder : 'asc'} onClick={()=>handleProbTableSortRequest('probability')}>Probability</TableSortLabel></TableCell>
                                                    </TableRow></TableHead>
                                                    <TableBody>
                                                        {sortedProbabilities.map(p => (
                                                            <TableRow key={p.className} hover selected={p.className === selectedPredictionDetailsJson.predicted_class_name}>
                                                                <TableCell>{p.className}</TableCell>
                                                                <TableCell align="right">{(p.probability * 100).toFixed(2)}%</TableCell>
                                                            </TableRow>
                                                        ))}
                                                    </TableBody>
                                                </Table>
                                            </TableContainer>
                                        </Grid>
                                        <Grid item xs={12} md={7}> {/* Probability Bar Plot */}
                                            <Typography variant="subtitle1" gutterBottom>Probability Distribution Plot</Typography>
                                            {predictionArtifacts.find(a => a.name === 'probability_distribution.png') ? (
                                                <PlotViewer
                                                    artifactUrl={getPlotUrl('probability_distribution.png')}
                                                    altText="Probability Distribution Plot"
                                                    onZoom={() => openFullscreen(getPlotUrl('probability_distribution.png'), 'Probability Distribution')}
                                                />
                                            ) : <Typography variant="caption">Plot not available.</Typography>}
                                        </Grid>
                                    </Grid>
                                )}
                            </Box>

                            {/* LIME Tab Panel */}
                            <Box role="tabpanel" hidden={activeContentTab !== 1} sx={{pt:2}}>
                                {activeContentTab === 1 && selectedPredictionDetailsJson.lime_explanation && !selectedPredictionDetailsJson.lime_explanation.error && (
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={5}> {/* LIME Weights/Features */}
                                            <Typography variant="subtitle1" gutterBottom>LIME Feature Weights for "{selectedPredictionDetailsJson.lime_explanation.explained_class_name}"</Typography>
                                            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 350 }}>
                                                <Table stickyHeader size="small">
                                                    <TableHead><TableRow><TableCell>Feature (Segment ID)</TableCell><TableCell align="right">Weight</TableCell></TableRow></TableHead>
                                                    <TableBody>
                                                        {(selectedPredictionDetailsJson.lime_explanation.feature_weights || []).map(fw => (
                                                            <TableRow key={fw[0]} hover>
                                                                <TableCell>{fw[0]}</TableCell>
                                                                <TableCell align="right" sx={{color: fw[1] > 0 ? 'success.main' : 'error.main'}}>{fw[1].toFixed(4)}</TableCell>
                                                            </TableRow>
                                                        ))}
                                                    </TableBody>
                                                </Table>
                                            </TableContainer>
                                            <Typography variant="caption" display="block" sx={{mt:1}}>Showing top {selectedPredictionDetailsJson.lime_explanation.num_features_from_lime_run} features.</Typography>
                                        </Grid>
                                        <Grid item xs={12} md={7}> {/* LIME Explanation Plot */}
                                            <Typography variant="subtitle1" gutterBottom>LIME Explanation Plot</Typography>
                                            {predictionArtifacts.find(a => a.name === 'lime_explanation.png') ? (
                                                <PlotViewer
                                                    artifactUrl={getPlotUrl('lime_explanation.png')}
                                                    altText="LIME Explanation Plot"
                                                    onZoom={() => openFullscreen(getPlotUrl('lime_explanation.png'), 'LIME Explanation')}
                                                />
                                            ) : <Typography variant="caption">LIME plot not available.</Typography>}
                                        </Grid>
                                    </Grid>
                                )}
                                {activeContentTab === 1 && selectedPredictionDetailsJson.lime_explanation?.error && (
                                    <Alert severity="warning">LIME explanation could not be generated: {selectedPredictionDetailsJson.lime_explanation.error}</Alert>
                                )}
                            </Box>
                        </Paper>
                    )}
                    {selectedPrediction && !selectedPredictionDetailsJson && !isLoadingSelectedPredictionContent && (
                        <Paper elevation={2} sx={{p:3, textAlign:'center'}}>
                            <Typography color="error">Could not load prediction details (JSON file not found or unparsable).</Typography>
                        </Paper>
                    )}
                </Grid>
            </Grid>

            <NewPredictionModal open={newPredictionModalOpen} onClose={() => setNewPredictionModalOpen(false)} imageId={Number(routeImageId)} onPredictionCreated={handlePredictionCreated} />
            <ImageFullscreenModal open={fullscreenModal.open} onClose={() => setFullscreenModal({open:false, src:null, title:''})} imageUrl={fullscreenModal.type === 'url' ? fullscreenModal.src : null} imageBlob={fullscreenModal.type === 'blob' ? fullscreenModal.src : null} title={fullscreenModal.title} />
        </Container>
    );
};

export default ViewImagePredictionsPage;