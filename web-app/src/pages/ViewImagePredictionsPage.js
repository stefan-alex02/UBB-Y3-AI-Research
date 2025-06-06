import React, {useCallback, useEffect, useState} from 'react';
import {useNavigate, useParams} from 'react-router-dom';
import {
    Alert,
    Box,
    Button,
    CardMedia,
    Chip,
    CircularProgress,
    Container,
    Divider,
    Grid,
    IconButton,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Paper,
    Skeleton,
    Typography
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import AssessmentIcon from '@mui/icons-material/Assessment';
import FolderIcon from '@mui/icons-material/Folder';
import ArticleIcon from '@mui/icons-material/Article';
import BrokenImageIcon from '@mui/icons-material/BrokenImage';
import ZoomInIcon from '@mui/icons-material/ZoomIn'; // For fullscreen image icon
import imageService from '../services/imageService';
import predictionService from '../services/predictionService';
import ArtifactViewer from '../components/ArtifactViewer/ArtifactViewer';
import LoadingSpinner from '../components/LoadingSpinner';
import useAuth from '../hooks/useAuth';
import ImageFullscreenModal from '../components/ImageFullscreenModal'; // Import the modal
import NewPredictionModal from '../components/Modals/NewPredictionModal';
import {API_BASE_URL} from "../config"; // Assuming modal is moved

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
    const { imageId } = useParams(); // imageId from URL is string, convert to Long if needed for API
    const navigate = useNavigate();
    const { user } = useAuth();

    const [imageDetails, setImageDetails] = useState(null); // ImageDTO from DB
    const [predictions, setPredictions] = useState([]);   // List<PredictionDTO>
    const [selectedPrediction, setSelectedPrediction] = useState(null); // Full PredictionDTO

    const [predictionArtifacts, setPredictionArtifacts] = useState([]); // List<ArtifactNode> for selected prediction
    const [selectedArtifactToView, setSelectedArtifactToView] = useState(null); // { name, type, content, url } for ArtifactViewer

    // State for the main uploaded image display
    const [mainImageBlob, setMainImageBlob] = useState(null);
    const [isLoadingMainImage, setIsLoadingMainImage] = useState(true);
    const [mainImageError, setMainImageError] = useState(false);

    // State for general page loading and errors
    const [isLoadingPageData, setIsLoadingPageData] = useState(true);
    const [pageError, setPageError] = useState(null);

    // State for loading specific artifact content
    const [isLoadingArtifactContent, setIsLoadingArtifactContent] = useState(false);

    // State for modals
    const [newPredictionModalOpen, setNewPredictionModalOpen] = useState(false);
    const [fullscreenModalOpen, setFullscreenModalOpen] = useState(false);
    const [fullscreenModalSource, setFullscreenModalSource] = useState({ src: null, type: 'url', title: '' });


    const fetchPageData = useCallback(async () => {
        if (!user || !imageId) return;
        setIsLoadingPageData(true);
        setPageError(null);
        setMainImageError(false);
        setIsLoadingMainImage(true);

        let tempMainImageBlob = null;

        try {
            // 1. Fetch image metadata
            const imgData = await imageService.getImageByIdForUser(Number(imageId), user.username);
            setImageDetails(imgData);

            // 2. Fetch image content (blob)
            if (imgData && imgData.id) {
                try {
                    tempMainImageBlob = await imageService.getImageContentBlob(imgData.id);
                    setMainImageBlob(tempMainImageBlob);
                } catch (contentError) {
                    console.error("Failed to load main image content:", contentError);
                    setMainImageError(true);
                }
            } else {
                setMainImageError(true); // No imgData to fetch content
            }

            // 3. Fetch predictions for this image
            const predData = await predictionService.getPredictionsForImage(Number(imageId));
            setPredictions(predData);
            if (predData.length > 0 && !selectedPrediction) { // Auto-select first if none selected
                handlePredictionSelect(predData[0]);
            } else if (selectedPrediction) { // If one was already selected, refresh its artifacts
                fetchPredictionArtifacts(selectedPrediction);
            }

        } catch (err) {
            setPageError(err.response?.data?.detail || err.message || 'Failed to load page data.');
            setMainImageError(true);
        } finally {
            setIsLoadingPageData(false);
            setIsLoadingMainImage(false); // Content loading attempt is done
        }
    }, [imageId, user, selectedPrediction]); // selectedPrediction added to refresh artifacts if it changes

    useEffect(() => {
        fetchPageData();
    }, [fetchPageData]); // Called once on mount and if dependencies change


    const fetchPredictionArtifacts = useCallback(async (prediction, subPath = '') => {
        if (!user || !prediction) return;
        setIsLoadingArtifactContent(true); // Use this for the artifact list loading
        setSelectedArtifactToView(null); // Clear currently viewed artifact
        try {
            const data = await predictionService.listPredictionArtifacts(
                user.username,
                String(prediction.imageId), // Ensure string for path construction
                prediction.modelExperimentRunId,
                subPath
            );
            setPredictionArtifacts(data);
        } catch (err) {
            setPageError(prev => `${prev ? prev + '; ' : ''}Failed to list prediction artifacts: ${err.message}`);
            setPredictionArtifacts([]);
        } finally {
            setIsLoadingArtifactContent(false);
        }
    }, [user]);

    const handlePredictionSelect = (prediction) => {
        setSelectedPrediction(prediction);
        setPredictionArtifacts([]); // Clear old artifacts before fetching new ones
        fetchPredictionArtifacts(prediction);
    };

    const handleArtifactClick = async (artifactNode) => {
        if (!user || !selectedPrediction || !imageDetails) return;

        setIsLoadingArtifactContent(true);
        setSelectedArtifactToView({ name: artifactNode.name, type: 'loading', content: null, url: null });
        const artifactType = getArtifactType(artifactNode.name);
        const artifactRelativePath = artifactNode.path; // e.g., "plots/lime.png" or "prediction_details.json"

        try {
            const pythonApiBaseForPredArtifacts = `/python-proxy-artifacts/predictions/${user.username}/${imageDetails.id}/${selectedPrediction.modelExperimentRunId}`;

            if (artifactType === 'image') {
                setSelectedArtifactToView({
                    name: artifactNode.name,
                    type: artifactType,
                    url: `${API_BASE_URL}${pythonApiBaseForPredArtifacts}/${artifactRelativePath}`,
                    content: null,
                });
            } else { // For JSON, log, csv
                const content = await predictionService.getPredictionArtifactContent(
                    user.username,
                    String(imageDetails.id),
                    selectedPrediction.modelExperimentRunId,
                    artifactRelativePath
                );
                setSelectedArtifactToView({
                    name: artifactNode.name,
                    type: artifactType,
                    content: content,
                    url: null,
                });
            }
        } catch (err) {
            setPageError(`Failed to load artifact ${artifactNode.name}: ${err.message}`);
            setSelectedArtifactToView({ name: artifactNode.name, type: 'error', content: err.message });
        } finally {
            setIsLoadingArtifactContent(false);
        }
    };

    const handlePredictionCreated = () => {
        setNewPredictionModalOpen(false);
        fetchPageData(); // Refresh predictions list and potentially image details
    };

    const openFullscreenImage = (blob, title) => {
        setFullscreenModalSource({ src: blob, type: 'blob', title: title });
        setFullscreenModalOpen(true);
    };
    const openFullscreenPlot = (url, title) => {
        setFullscreenModalSource({ src: url, type: 'url', title: title });
        setFullscreenModalOpen(true);
    };


    if (isLoadingPageData && !imageDetails) return <Container sx={{mt:2}}><LoadingSpinner /></Container>;
    if (pageError && !imageDetails) return <Container sx={{mt:2}}><Alert severity="error">{pageError}</Alert></Container>;

    return (
        <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
            <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/images')} sx={{ mb: 2 }}>
                Back to Images
            </Button>

            <Grid container spacing={3}>
                {/* Left Column: Image and Predictions List */}
                <Grid item xs={12} md={4}>
                    {imageDetails && (
                        <Paper elevation={3} sx={{ p: 2, mb: 2, position: 'sticky', top: '80px' }}>
                            <Typography variant="h5" gutterBottom>Image: {imageDetails.id}.{imageDetails.format}</Typography>
                            <Box sx={{ textAlign: 'center', mb: 1, position: 'relative' }}>
                                {isLoadingMainImage ? (
                                    <Skeleton variant="rectangular" width="100%" height={250} animation="wave" sx={{ borderRadius: 1 }} />
                                ) : mainImageError ? (
                                    <Box sx={{ height: 250, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: 'grey.200', borderRadius: 1 }}>
                                        <BrokenImageIcon color="action" sx={{ fontSize: 60 }} />
                                        <Typography color="textSecondary">Image Preview Unavailable</Typography>
                                    </Box>
                                ) : (
                                    <>
                                        <CardMedia component="img" image={mainImageBlob ? URL.createObjectURL(mainImageBlob) : ''} alt={`Image ${imageDetails.id}`} sx={{ borderRadius: 1, maxHeight: 300, width: 'auto', maxWidth: '100%', objectFit: 'contain' }} />
                                        <IconButton onClick={() => mainImageBlob && openFullscreenImage(mainImageBlob, `Image ${imageDetails.id}.${imageDetails.format}`)}
                                                    sx={{position:'absolute', top: 8, right: 8, backgroundColor: 'rgba(0,0,0,0.3)', '&:hover': {backgroundColor: 'rgba(0,0,0,0.5)'}}}
                                                    size="small"
                                                    disabled={!mainImageBlob || mainImageError}
                                        >
                                            <ZoomInIcon sx={{color: 'white'}}/>
                                        </IconButton>
                                    </>
                                )}
                            </Box>
                            <Typography variant="body2" color="text.secondary">Uploaded: {new Date(imageDetails.uploadedAt).toLocaleString()}</Typography>
                            <Button fullWidth variant="contained" startIcon={<AddCircleOutlineIcon />} onClick={() => setNewPredictionModalOpen(true)} sx={{ mt: 2 }}>
                                New Prediction
                            </Button>
                        </Paper>
                    )}

                    <Paper elevation={1} sx={{p:1, mt:2}}>
                        <Typography variant="h6" sx={{p:1}}>Predictions History</Typography>
                        {isLoadingPageData && predictions.length === 0 ? <CircularProgress sx={{m:2}} /> : (
                            <List dense sx={{maxHeight: 'calc(100vh - 500px)', overflowY: 'auto'}}> {/* Adjust maxHeight as needed */}
                                {predictions.map(pred => (
                                    <ListItemButton key={pred.id} selected={selectedPrediction?.id === pred.id} onClick={() => handlePredictionSelect(pred)}>
                                        <ListItemIcon sx={{minWidth: '36px'}}><AssessmentIcon fontSize="small" color={selectedPrediction?.id === pred.id ? "primary" : "action"}/></ListItemIcon>
                                        <ListItemText
                                            primary={`${pred.predictedClass} (${(pred.confidence * 100).toFixed(1)}%)`}
                                            secondary={`Model: ...${pred.modelExperimentRunId.slice(-6)} | ${new Date(pred.predictionTimestamp).toLocaleDateString()}`}
                                            primaryTypographyProps={{variant: 'body2', fontWeight: selectedPrediction?.id === pred.id ? 'bold' : 'normal'}}
                                        />
                                    </ListItemButton>
                                ))}
                                {predictions.length === 0 && !isLoadingPageData && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No predictions yet for this image.</Typography>}
                            </List>
                        )}
                    </Paper>
                </Grid>

                {/* Right Column: Selected Prediction Details & Artifacts */}
                <Grid item xs={12} md={8}>
                    {selectedPrediction ? (
                        <Paper elevation={2} sx={{ p: 2, minHeight: 'calc(100vh - 150px)' /* Example height */ }}>
                            <Typography variant="h5" gutterBottom>
                                Prediction Details
                            </Typography>
                            <Typography variant="h6">Predicted Class: <Chip label={selectedPrediction.predictedClass} color="primary" size="small"/></Typography>
                            <Typography variant="subtitle1">Confidence: {(selectedPrediction.confidence * 100).toFixed(2)}%</Typography>
                            <Typography variant="body2" color="text.secondary">Model from Experiment ID: {selectedPrediction.modelExperimentRunId}</Typography>
                            <Typography variant="body2" color="text.secondary">Prediction Timestamp: {new Date(selectedPrediction.predictionTimestamp).toLocaleString()}</Typography>
                            <Divider sx={{ my: 2 }} />

                            <Typography variant="h6">Artifacts for this Prediction</Typography>
                            {/* TODO: Add Breadcrumbs for prediction artifacts if they can have sub-folders (currently they don't in the design) */}
                            {isLoadingArtifactContent && predictionArtifacts.length === 0 ? <CircularProgress sx={{my:2}} /> : (
                                <List dense component={Paper} elevation={0} sx={{border: '1px solid #eee', borderRadius:1, maxHeight: 150, overflowY:'auto', mb:1}}>
                                    {predictionArtifacts.sort((a,b) => a.name.localeCompare(b.name)).map((node) => ( // Simple sort
                                        <ListItem key={node.path} disablePadding >
                                            <ListItemButton onClick={() => handleArtifactClick(node)} selected={selectedArtifactToView?.name === node.name}>
                                                <ListItemIcon sx={{minWidth: '32px'}}>{node.type === 'folder' ? <FolderIcon fontSize="small" /> : <ArticleIcon fontSize="small" />}</ListItemIcon>
                                                <ListItemText primary={node.name} primaryTypographyProps={{ variant: 'body2', noWrap: true }} />
                                                {getArtifactType(node.name) === 'image' && (
                                                    <IconButton size="small" onClick={(e) => {
                                                        e.stopPropagation(); // Prevent ListItemButton click
                                                        const artifactBasePath = `${API_BASE_URL}/python-proxy-artifacts/predictions/${user.username}/${imageDetails.id}/${selectedPrediction.modelExperimentRunId}`;
                                                        openFullscreenPlot(`${artifactBasePath}/${node.path}`, `${node.name} for Prediction ${selectedPrediction.id}`);
                                                    }}>
                                                        <ZoomInIcon fontSize="small"/>
                                                    </IconButton>
                                                )}
                                            </ListItemButton>
                                        </ListItem>
                                    ))}
                                    {predictionArtifacts.length === 0 && !isLoadingArtifactContent && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No artifacts found for this prediction.</Typography>}
                                </List>
                            )}
                            <Box sx={{ mt: 1, border: '1px dashed grey', p: selectedArtifactToView ? 0 : 2, borderRadius: 1, minHeight: 300, maxHeight: 'calc(100vh - 550px)', overflowY:'auto' }}>
                                {isLoadingArtifactContent && selectedArtifactToView?.type === 'loading' && <LoadingSpinner />}
                                {selectedArtifactToView && selectedArtifactToView.type !== 'loading' && (
                                    <ArtifactViewer
                                        artifactName={selectedArtifactToView.name}
                                        artifactType={selectedArtifactToView.type}
                                        artifactContent={selectedArtifactToView.content}
                                        artifactUrl={selectedArtifactToView.url} // Used if type is 'image'
                                        title={selectedArtifactToView.name}
                                    />
                                )}
                                {!selectedArtifactToView && !isLoadingArtifactContent && (
                                    pageError ? <Alert severity="warning" sx={{m:1}}>{pageError.includes("Failed to load artifact") ? pageError : "Select an artifact to view."}</Alert> :
                                        <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: '20%' }}>
                                            Select an artifact to view its content.
                                        </Typography>
                                )}
                            </Box>
                        </Paper>
                    ) : (
                        <Paper elevation={2} sx={{ p: 3, textAlign: 'center', minHeight: 'calc(70vh)', display:'flex', flexDirection:'column', justifyContent:'center' }}>
                            <Typography variant="h6" color="text.secondary">
                                {isLoadingPageData ? 'Loading predictions...' : 'Select a prediction from the list on the left, or create a new one.'}
                            </Typography>
                        </Paper>
                    )}
                </Grid>
            </Grid>

            {imageDetails && (
                <NewPredictionModal
                    open={newPredictionModalOpen}
                    onClose={() => setNewPredictionModalOpen(false)}
                    imageId={Number(imageId)} // Pass the imageId
                    onPredictionCreated={handlePredictionCreated}
                />
            )}
            <ImageFullscreenModal
                open={fullscreenModalOpen}
                onClose={() => setFullscreenModalOpen(false)}
                imageUrl={fullscreenModalSource.type === 'url' ? fullscreenModalSource.src : null}
                imageBlob={fullscreenModalSource.type === 'blob' ? fullscreenModalSource.src : null}
                title={fullscreenModalSource.title}
            />
        </Container>
    );
};

export default ViewImagePredictionsPage;