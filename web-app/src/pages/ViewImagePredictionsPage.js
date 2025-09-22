import React, {useCallback, useEffect, useMemo, useState} from 'react';
import {useNavigate, useParams} from 'react-router-dom';
import {
    Alert,
    Box,
    Button,
    CardMedia,
    Chip,
    CircularProgress,
    Collapse,
    Container,
    FormControl,
    Grid,
    IconButton,
    InputLabel,
    List,
    ListItemButton,
    MenuItem,
    Paper,
    Select,
    Slider,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TableSortLabel,
    Tabs,
    TextField,
    Tooltip,
    Typography,
    useMediaQuery,
    useTheme
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import AssessmentIcon from '@mui/icons-material/Assessment';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import BarChartIcon from '@mui/icons-material/BarChart';
import imageService from '../services/imageService';
import predictionService from '../services/predictionService';
import PlotViewer from '../components/ArtifactViewer/PlotViewer';
import NewPredictionModal from '../components/Modals/NewPredictionModal';
import InsightsIcon from '@mui/icons-material/Insights';
import {getComparator, stableSort} from "../utils/tableUtils";
import {formatDateSafe} from "../utils/dateUtils";
import useAuth from "../hooks/useAuth";
import ImageFullscreenModal from "../components/Modals/ImageFullscreenModal";
import FilterListIcon from "@mui/icons-material/FilterList";
import RefreshIcon from "@mui/icons-material/Refresh";
import {DATASET_NAMES, MODEL_TYPES} from "./experimentConfig";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import DatasetIcon from "@mui/icons-material/Dataset";
import DeleteIcon from "@mui/icons-material/Delete";
import ConfirmDialog from "../components/ConfirmDialog";

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
    const [mainImageBlob, setMainImageBlob] = useState(null);
    const [isLoadingMainImage, setIsLoadingMainImage] = useState(true);
    const [mainImageError, setMainImageError] = useState(false);

    const [predictions, setPredictions] = useState([]);
    const [selectedPrediction, setSelectedPrediction] = useState(null);

    const [selectedPredictionDetailsJson, setSelectedPredictionDetailsJson] = useState(null);
    const [probabilityPlotUrl, setProbabilityPlotUrl] = useState(null);
    const [limePlotUrl, setLimePlotUrl] = useState(null);
    const [predictionArtifacts, setPredictionArtifacts] = useState([]);

    const [activeContentTab, setActiveContentTab] = useState(0);

    const [newPredictionModalOpen, setNewPredictionModalOpen] = useState(false);
    const [fullscreenModal, setFullscreenModal] = useState({ open: false, src: null, title: '', type: 'url' });

    const [predictionFilters, setPredictionFilters] = useState({
        experimentNameContains: '',
        predictedClassContains: '',
        modelType: '',
        confidenceOver: 0,
    });
    const [showPredictionFilters, setShowPredictionFilters] = useState(false);

    const [isLoadingPageData, setIsLoadingPageData] = useState(true);
    const [isLoadingPredictionsList, setIsLoadingPredictionsList] = useState(false);
    const [isLoadingSelectedPredictionContent, setIsLoadingSelectedPredictionContent] = useState(false);
    const [pageError, setPageError] = useState(null);

    const theme = useTheme();
    const isMobileOrSmallScreen = useMediaQuery(theme.breakpoints.down('md'));

    const [probTableOrder, setProbTableOrder] = useState('desc');
    const [probTableOrderBy, setProbTableOrderBy] = useState('probability');

    const [limeTableOrder, setLimeTableOrder] = useState('desc');
    const [limeTableOrderBy, setLimeTableOrderBy] = useState('weight');

    const [isLoadingProbPlot, setIsLoadingProbPlot] = useState(false);
    const [isLoadingLimePlot, setIsLoadingLimePlot] = useState(false);

    const isDesktopLayout = useMediaQuery(theme.breakpoints.up('md'));
    const headerAndPaddingHeight = theme.mixins.toolbar.minHeight + theme.spacing(2 + 2 + 3);

    const [deletePredictionConfirm, setDeletePredictionConfirm] = useState({ open: false, prediction: null });

    const fetchPageInitialData = useCallback(async () => {
        if (!user || !routeImageId) return;
        setIsLoadingPageData(true); setIsLoadingMainImage(true); setMainImageError(false); setPageError(null);
        try {
            const imgData = await imageService.getImageByIdForUser(Number(routeImageId), user.username);
            setImageDetails(imgData);
            if (imgData?.id) {
                const blob = await imageService.getImageContentBlob(imgData.id);
                setMainImageBlob(blob);
            } else { setMainImageError(true); }
        } catch (err) {
            setPageError(err.response?.data?.message || "Failed to load image details.");
            setMainImageError(true);
        } finally { setIsLoadingMainImage(false); }

        try {
            setIsLoadingPredictionsList(true);
            const predData = await predictionService.getPredictionsForImage(Number(routeImageId));
            setPredictions(predData);
        } catch (err) {
            setPageError(prev => `${prev ? prev + '; ' : ''}Failed to load predictions list.`);
        } finally { setIsLoadingPredictionsList(false); setIsLoadingPageData(false); }
    }, [routeImageId, user]);

    useEffect(() => { fetchPageInitialData(); }, [fetchPageInitialData]);

    const fetchSelectedPredictionData = useCallback(async (prediction) => {
        if (!user || !prediction || !imageDetails) {
            setSelectedPredictionDetailsJson(null);
            setProbabilityPlotUrl(p => { if(p) URL.revokeObjectURL(p); return null; });
            setLimePlotUrl(l => { if(l) URL.revokeObjectURL(l); return null; });
            setPredictionArtifacts([]);
            return;
        }
        setIsLoadingSelectedPredictionContent(true);
        setPageError(null);
        setSelectedPredictionDetailsJson(null);
        setProbabilityPlotUrl(p => { if(p) URL.revokeObjectURL(p); return null; });
        setLimePlotUrl(l => { if(l) URL.revokeObjectURL(l); return null; });
        let fetchedPlotArtifacts = [];

        try {
            const jsonContentStr = await predictionService.getPredictionArtifactContent(
                String(prediction.id), "prediction_details.json"
            );
            const parsedJson = JSON.parse(jsonContentStr);
            setSelectedPredictionDetailsJson(parsedJson);

            fetchedPlotArtifacts = await predictionService.listPredictionArtifacts(
                String(prediction.id), "plots"
            );
            setPredictionArtifacts(fetchedPlotArtifacts);

            const probabilityPlotNode = fetchedPlotArtifacts.find(a => a.name === 'probability_distribution.png' && getArtifactType(a.name) === 'image');
            if (probabilityPlotNode) {
                setIsLoadingProbPlot(true);
                try {
                    const blob = await predictionService.getPredictionArtifactContent(
                        String(prediction.id), probabilityPlotNode.path
                    );
                    if (blob) setProbabilityPlotUrl(URL.createObjectURL(blob));
                } catch (plotError) {
                    console.error("Probability plot content not found/error:", plotError);
                    setProbabilityPlotUrl(null);
                } finally {
                    setIsLoadingProbPlot(false);
                }
            } else {
                setProbabilityPlotUrl(null);
            }

            if (parsedJson?.lime_explanation && !parsedJson.lime_explanation.error) {
                const limePlotNode = fetchedPlotArtifacts.find(a => a.name === 'lime_explanation.png' && getArtifactType(a.name) === 'image');
                if (limePlotNode) {
                    setIsLoadingLimePlot(true);
                    try {
                        const blob = await predictionService.getPredictionArtifactContent(
                            String(prediction.id), limePlotNode.path
                        );
                        if (blob) setLimePlotUrl(URL.createObjectURL(blob));
                    } catch (limePlotError) {
                        console.error("LIME plot content not found/error:", limePlotError);
                        setLimePlotUrl(null);
                    } finally {
                        setIsLoadingLimePlot(false);
                    }
                } else {
                    setLimePlotUrl(null);
                }
            } else {
                setLimePlotUrl(null);
            }

        } catch (err) {
            setPageError(err.response?.data?.message || err.message || `Failed to load details for prediction made with model ${prediction.model_experiment_run_id}.`);
            setSelectedPredictionDetailsJson({ error: "Could not load prediction details." });
        } finally {
            setIsLoadingSelectedPredictionContent(false);
        }
    }, [user, imageDetails]);

    const handlePredictionSelect = (prediction) => {
        if (selectedPrediction?.id === prediction?.id) return;
        setSelectedPrediction(prediction);
        setActiveContentTab(0);
        if (prediction) {
            fetchSelectedPredictionData(prediction);
        } else {
            setSelectedPredictionDetailsJson(null);
            setProbabilityPlotUrl(p => { if(p) URL.revokeObjectURL(p); return null; });
            setLimePlotUrl(l => { if(l) URL.revokeObjectURL(l); return null; });
        }
    };

    useEffect(() => { const url = probabilityPlotUrl; return () => { if (url) URL.revokeObjectURL(url); }; }, [probabilityPlotUrl]);
    useEffect(() => { const url = limePlotUrl; return () => { if (url) URL.revokeObjectURL(url); }; }, [limePlotUrl]);

    const mainImageDisplayUrl = useMemo(() => mainImageBlob ? URL.createObjectURL(mainImageBlob) : null, [mainImageBlob]);
    useEffect(() => { return () => { if (mainImageDisplayUrl) URL.revokeObjectURL(mainImageDisplayUrl); };}, [mainImageDisplayUrl]);


    const openFullscreen = (srcOrBlob, title, type = 'url') => {
        setFullscreenModal({ open: true, src: srcOrBlob, title, type });
    };

    const handleManualRefreshPredictions = () => {
        setSelectedPrediction(null);
        setSelectedPredictionDetailsJson(null);
        setProbabilityPlotUrl(p => { if(p) URL.revokeObjectURL(p); return null; });
        setLimePlotUrl(l => { if(l) URL.revokeObjectURL(l); return null; });
        fetchPageInitialData(true);
    };

    const handleNewPredictionClick = () => {
        setSelectedPrediction(null);
        setSelectedPredictionDetailsJson(null);
        setProbabilityPlotUrl(null);
        setLimePlotUrl(null);
        setNewPredictionModalOpen(true);
    };

    const handlePredictionCreated = () => {
        setNewPredictionModalOpen(false);
        setSelectedPrediction(null);
        setSelectedPredictionDetailsJson(null);
        setProbabilityPlotUrl(null);
        setLimePlotUrl(null);
        fetchPageInitialData();
    };

    const filteredPredictions = useMemo(() => {
        return predictions.filter(p => {
            const modelExperimentIdentifier = p.model_experiment_name || p.model_experiment_run_id || "";
            const nameMatch = !predictionFilters.experimentNameContains ||
                modelExperimentIdentifier.toLowerCase().includes(predictionFilters.experimentNameContains.toLowerCase());

            const classMatch = !predictionFilters.predictedClassContains ||
                (p.predicted_class && p.predicted_class.toLowerCase().includes(predictionFilters.predictedClassContains.toLowerCase()));

            const typeMatch = !predictionFilters.modelType || p.model_type === predictionFilters.modelType;

            const datasetMatch = !predictionFilters.datasetName ||
                (p.dataset_name && p.dataset_name === predictionFilters.datasetName);

            const confidenceMatch = p.confidence * 100 >= predictionFilters.confidenceOver;

            return nameMatch && classMatch && typeMatch && datasetMatch && confidenceMatch;
        });
    }, [predictions, predictionFilters]);

    const handlePredictionFilterChange = (event) => {
        const { name, value } = event.target;
        setPredictionFilters(prev => ({ ...prev, [name]: value }));
    };

    const handleConfidenceFilterChange = (event, newValue) => {
        setPredictionFilters(prev => ({ ...prev, confidenceOver: newValue }));
    };

    const handleProbTableSortRequest = (property) => {
        const isAsc = probTableOrderBy === property && probTableOrder === 'asc';
        setProbTableOrder(isAsc ? 'desc' : 'asc');
        setProbTableOrderBy(property);
    };

    const sortedProbabilities = useMemo(() => {
        if (!selectedPredictionDetailsJson || !selectedPredictionDetailsJson.top_k_predictions_for_plot) {
            return [];
        }
        const data = selectedPredictionDetailsJson.top_k_predictions_for_plot.map(p => ({ className: p[0], probability: p[1] }));
        return stableSort(data, getComparator(probTableOrder, probTableOrderBy));
    }, [selectedPredictionDetailsJson, probTableOrder, probTableOrderBy]);

    const handleLimeTableSortRequest = (property) => {
        const isAsc = limeTableOrderBy === property && limeTableOrder === 'asc';
        setLimeTableOrder(isAsc ? 'desc' : 'asc');
        setLimeTableOrderBy(property);
    };

    const sortedLimeWeights = useMemo(() => {
        if (!selectedPredictionDetailsJson?.lime_explanation?.feature_weights) {
            return [];
        }
        const data = selectedPredictionDetailsJson.lime_explanation.feature_weights.map(fw => ({
            segmentId: fw[0],
            weight: fw[1]
        }));

        const limeComparator = (order, orderBy) => {
            return order === 'desc'
                ? (a, b) => (b[orderBy] < a[orderBy] ? -1 : (b[orderBy] > a[orderBy] ? 1 : 0))
                : (a, b) => -(b[orderBy] < a[orderBy] ? -1 : (b[orderBy] > a[orderBy] ? 1 : 0));
        };
        return stableSort(data, limeComparator(limeTableOrder, limeTableOrderBy));
    }, [selectedPredictionDetailsJson, limeTableOrder, limeTableOrderBy]);

    const openDeletePredictionDialog = (predictionToDelete) => {
        if (predictionToDelete) {
            setDeletePredictionConfirm({ open: true, prediction: predictionToDelete });
        }
    };

    const handleConfirmDeletePrediction = async () => {
        if (deletePredictionConfirm.prediction) {
            setPageError(null);
            try {
                await predictionService.deletePrediction(deletePredictionConfirm.prediction.id);
                setDeletePredictionConfirm({ open: false, prediction: null });
                fetchPageInitialData();
                setSelectedPrediction(null);
                setSelectedPredictionDetailsJson(null);
                setProbabilityPlotUrl(null);
                setLimePlotUrl(null);
            } catch (err) {
                setPageError(err.response?.data?.message || `Failed to delete prediction.`);
                setDeletePredictionConfirm({ open: false, prediction: null });
            }
        }
    };

    if (isLoadingPageData && !imageDetails) return <Container sx={{mt:2, textAlign: 'center'}}><CircularProgress size={50} sx={{mt:5}}/><Typography>Loading image details...</Typography></Container>;
    if (pageError && !imageDetails) return <Container sx={{mt:2}}><Alert severity="error" onClose={() => setPageError(null)}>{pageError}</Alert></Container>;
    if (!imageDetails) return <Container sx={{mt:2}}><Alert severity="info">Image details not available or ID not found.</Alert></Container>;


    return (
        <Container
            maxWidth="xl"
            sx={{
                mt: 2, mb: 4,
                display: 'flex', flexDirection: 'column',
                height: `calc(100vh - ${headerAndPaddingHeight}px)`,
                overflow: 'hidden',
            }}
        >
            <Button
                startIcon={<ArrowBackIcon />}
                onClick={() => navigate('/images')}
                sx={{
                    mb: 2,
                    flexShrink: 0,
                    width: 'auto',
                    alignSelf: 'flex-start'
                }}
            >
                Back to Images
            </Button>

            <Grid container spacing={isDesktopLayout ? 3 : 2} sx={{
                flexGrow: 1,
                overflow: 'hidden',
                width: '100%',
                flexWrap: {
                    xs: 'wrap',
                    md: 'nowrap'
                }
            }}>
                {/* --- LEFT COLUMN --- */}
                <Grid
                    item
                    xs={12}
                    md={5}
                    lg={4}
                    sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        height: isDesktopLayout ? '100%' : 'auto',
                        overflowY: isDesktopLayout ? 'auto' : 'visible',
                        pr: isDesktopLayout ? 1 : 0,
                        // Update width constraints to match new grid sizes
                        width: isDesktopLayout ? { md: '41.667%', lg: '33.333%' } : '100%',
                        maxWidth: isDesktopLayout ? { md: '41.667%', lg: '33.333%' } : '100%',
                        flexBasis: isDesktopLayout ? { md: '41.667%', lg: '33.333%' } : '100%',
                        overflow: 'hidden',
                    }}
                >
                    {/* Image Details Paper */}
                    {imageDetails && (
                        <Paper elevation={3} sx={{ p: 2, mb: 2, flexShrink: 0 }}>
                            <Typography variant={isDesktopLayout ? "h6" : "h5"} gutterBottom>
                                Image: {imageDetails.id}.{imageDetails.format}
                            </Typography>
                            <Box sx={{ textAlign: 'center', mb: 1, position: 'relative' }}>
                                <CardMedia
                                    component="img"
                                    image={mainImageDisplayUrl}
                                    alt={`Image ${imageDetails.id}`}
                                    sx={{
                                        maxHeight: isDesktopLayout ? 260 : 280,
                                        objectFit: 'contain',
                                        borderRadius: 1,
                                        bgcolor: 'action.hover'
                                    }}
                                />
                                <IconButton onClick={() => mainImageBlob && openFullscreen(mainImageBlob, `Image ${imageDetails.id}`, 'blob')} sx={{position:'absolute', top: 8, right: 8, bgcolor: 'rgba(0,0,0,0.3)', '&:hover':{bgcolor:'rgba(0,0,0,0.5)'}}} size="small" disabled={!mainImageBlob || mainImageError}>
                                    <ZoomInIcon sx={{color: 'white'}}/>
                                </IconButton>
                            </Box>
                            <Typography variant="body2" color="text.secondary" sx={{textAlign:'center'}}>
                                Uploaded: {formatDateSafe(imageDetails.uploaded_at)}
                            </Typography>
                            <Button color="secondary" fullWidth variant="contained" startIcon={<AddCircleOutlineIcon />} onClick={handleNewPredictionClick} sx={{ mt: 2 }}>
                                New Prediction
                            </Button>
                        </Paper>
                    )}

                    {/* Predictions History Paper */}
                    <Paper elevation={1} sx={{ p:1, mt: isDesktopLayout ? 0 : 2, flexGrow: 1, display: 'flex', flexDirection: 'column', overflow:'hidden' }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1 }}>
                            <Typography variant="h6">Predictions History</Typography>
                            <Box>
                                <Tooltip title="Refresh Predictions List">
                                    <IconButton onClick={handleManualRefreshPredictions} size="small" sx={{mr:1}} disabled={isLoadingPredictionsList}>
                                        {isLoadingPredictionsList ? <CircularProgress size={20}/> : <RefreshIcon />}
                                    </IconButton>
                                </Tooltip>
                                <Tooltip title={showPredictionFilters ? "Hide Filters" : "Show Filters"}>
                                    <IconButton onClick={() => setShowPredictionFilters(f => !f)} size="small">
                                        <FilterListIcon color={showPredictionFilters ? "primary" : "action"} />
                                    </IconButton>
                                </Tooltip>
                            </Box>
                        </Box>
                        <Collapse in={showPredictionFilters} sx={{ flexShrink: 0 }}>
                            <Box sx={{p: 1, borderTop: '1px solid #eee', borderBottom: '1px solid #eee', mb: 1}}>
                                <Grid container spacing={1} sx={{ mb: 1 }}>
                                    <Grid item xs={6} sm={6}>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            label="Exp. Name"
                                            name="experimentNameContains"
                                            value={predictionFilters.experimentNameContains}
                                            onChange={handlePredictionFilterChange}
                                            sx={{ maxWidth: '140px' }}
                                        />
                                    </Grid>
                                    <Grid item xs={6} sm={6}>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            label="Class"
                                            name="predictedClassContains"
                                            value={predictionFilters.predictedClassContains}
                                            onChange={handlePredictionFilterChange}
                                            sx={{ maxWidth: '140px' }}
                                        />
                                    </Grid>
                                </Grid>

                                <Grid container spacing={1}>
                                    <Grid item xs={12} sm={4}>
                                        <FormControl fullWidth size="small">
                                            <InputLabel>Model Type</InputLabel>
                                            <Select
                                                name="modelType"
                                                value={predictionFilters.modelType}
                                                label="Model Type"
                                                onChange={handlePredictionFilterChange}
                                                sx={{ minWidth: '120px' }}
                                            >
                                                <MenuItem value=""><em>Any</em></MenuItem>
                                                {MODEL_TYPES.map(mt => <MenuItem key={mt.value} value={mt.value}>{mt.label}</MenuItem>)}
                                            </Select>
                                        </FormControl>
                                    </Grid>
                                    <Grid item xs={12} sm={4}>
                                        <FormControl fullWidth size="small">
                                            <InputLabel>Dataset Name</InputLabel>
                                            <Select
                                                name="datasetName"
                                                value={predictionFilters.datasetName}
                                                label="Dataset Name"
                                                onChange={handlePredictionFilterChange}
                                                sx={{ minWidth: '120px' }}
                                            >
                                                <MenuItem value=""><em>Any</em></MenuItem>
                                                {DATASET_NAMES.map(dn => <MenuItem key={dn} value={dn}>{dn}</MenuItem>)}
                                            </Select>
                                        </FormControl>
                                    </Grid>
                                    <Grid item xs={12} sm={4}>
                                        <Box sx={{px: 0.5}}>
                                            <Typography variant="caption" id="confidence-slider-label" sx={{ mb: 0, display: 'block' }}>
                                                Confidence Over: {predictionFilters.confidenceOver}%
                                            </Typography>
                                            <Slider
                                                name="confidenceOver"
                                                value={predictionFilters.confidenceOver}
                                                onChange={handleConfidenceFilterChange}
                                                aria-labelledby="confidence-slider-label"
                                                valueLabelDisplay="auto"
                                                step={1}
                                                min={0}
                                                max={100}
                                                size="small"
                                            />
                                        </Box>
                                    </Grid>
                                </Grid>
                            </Box>
                        </Collapse>
                        {/* Predictions List */}
                        <Box sx={{ flexGrow: 1, overflowY: 'auto', minHeight: 150 }}>
                            {isLoadingPredictionsList && predictions.length === 0 ? <CircularProgress sx={{m:2}} /> : (
                                <List dense sx={{py:0}}>
                                    {filteredPredictions.length === 0 && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No predictions match filters or none exist.</Typography>}
                                    {filteredPredictions.map(pred => (
                                        <ListItemButton
                                            key={pred.id}
                                            selected={selectedPrediction?.id === pred.id}
                                            onClick={() => handlePredictionSelect(pred)}
                                            sx={{ py: 0.75, display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}
                                        >
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                                                <Typography
                                                    variant="subtitle2"
                                                    component="div"
                                                    sx={{
                                                        fontWeight: selectedPrediction?.id === pred.id ? 'bold' : 500,
                                                        color: selectedPrediction?.id === pred.id ? 'primary.main' : 'text.primary',
                                                        display: 'flex',
                                                        alignItems: 'center'
                                                    }}
                                                >
                                                    <AssessmentIcon fontSize="inherit" sx={{ mr: 0.75, opacity:0.8 }} />
                                                    {pred.predicted_class || "N/A"} ({pred.confidence !== null && pred.confidence !== undefined ? (pred.confidence * 100).toFixed(1) : 'N/A'}%)
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary" sx={{ml:1, whiteSpace:'nowrap'}}>
                                                    {formatDateSafe(pred.prediction_timestamp, 'MMM d, HH:mm')}
                                                </Typography>
                                            </Box>
                                            <Box component="div" sx={{ fontSize: '0.7rem', color: 'text.secondary', mt: 0.25, width: '100%', pl: '28px' }}>
                                                <Tooltip title={`Experiment: ${pred.model_experiment_name || 'Unknown'}`}>
                                                    <Typography variant="caption" component="span" sx={{ display: 'inline-flex', alignItems: 'center', mr: 1, maxWidth: '40%', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>
                                                        <InsightsIcon fontSize="inherit" sx={{ mr: 0.5, opacity: 0.7 }} />
                                                        {pred.model_experiment_name || (pred.model_experiment_run_id && typeof pred.model_experiment_run_id === 'string' ? `...${pred.model_experiment_run_id.slice(-6)}` : 'Unknown')}
                                                    </Typography>
                                                </Tooltip>
                                                <Tooltip title={`Model Type: ${pred.model_type || 'N/A'}`}>
                                                    <Typography variant="caption" component="span" sx={{ display: 'inline-flex', alignItems: 'center', mr: 1}}>
                                                        <ModelTrainingIcon fontSize="inherit" sx={{ mr: 0.5, opacity: 0.7 }} /> {pred.model_type || 'N/A'}
                                                    </Typography>
                                                </Tooltip>
                                                <Tooltip title={`Dataset: ${pred.dataset_name || 'N/A'}`}>
                                                    <Typography variant="caption" component="span" sx={{ display: 'inline-flex', alignItems: 'center'}}>
                                                        <DatasetIcon fontSize="inherit" sx={{ mr: 0.5, opacity: 0.7 }} /> {pred.dataset_name || 'N/A'}
                                                    </Typography>
                                                </Tooltip>
                                            </Box>
                                        </ListItemButton>
                                    ))}
                                </List>
                            )}
                        </Box>
                    </Paper>
                </Grid>

                {/* --- RIGHT COLUMN --- */}
                <Grid
                    item
                    xs={12}
                    md={7}
                    lg={8}
                    sx={{
                        height: isDesktopLayout ? '100%' : 'auto',
                        overflowY: isDesktopLayout ? 'auto' : 'visible',
                        pl: isDesktopLayout ? 1 : 0,
                        maxWidth: isDesktopLayout ? { md: '58.333%', lg: '66.667%' } : '100%',
                        overflow: 'hidden',
                    }}
                >
                    {!selectedPrediction && !isLoadingSelectedPredictionContent && (
                        <Paper elevation={2} sx={{ p: 3, textAlign: 'center', minHeight: isMobileOrSmallScreen ? 200 : 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Typography variant="h6" color="text.secondary">
                                {predictions.length > 0 ? "Select a prediction from the list to view details." : "No predictions available for this image yet."}
                            </Typography>
                        </Paper>
                    )}
                    {isLoadingSelectedPredictionContent && (
                        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: isMobileOrSmallScreen ? 200 : 400 }}>
                            <CircularProgress />
                        </Box>
                    )}

                    {selectedPrediction && selectedPredictionDetailsJson && !isLoadingSelectedPredictionContent && (
                        <Paper elevation={2} sx={{ p: {xs:1, sm:2}, height: '100%', display: 'flex', flexDirection: 'column', overflow:'hidden' }}>
                            <Box sx={{mb:2, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                <Box>
                                    <Typography variant="h5" gutterBottom>
                                        Details for Prediction with Model: {selectedPrediction.model_experiment_name || `...${selectedPrediction.model_experiment_run_id.slice(-6)}`}
                                    </Typography>
                                    <Typography variant="h6">Main Prediction: <Chip label={`${selectedPredictionDetailsJson.predicted_class_name} (${(selectedPredictionDetailsJson.confidence * 100).toFixed(1)}%)`} color="primary"
                                                                                    clickable={false} onClick={() => {}} /></Typography>
                                    <Typography variant="caption" display="block">Image Source Path: {selectedPredictionDetailsJson.image_user_source_path}</Typography>
                                </Box>
                                <Tooltip title="Delete this Prediction">
                                    <IconButton onClick={() => openDeletePredictionDialog(selectedPrediction)} color="error" size="small">
                                        <DeleteIcon />
                                    </IconButton>
                                </Tooltip>
                            </Box>

                            <Tabs value={activeContentTab} onChange={(e, newValue) => setActiveContentTab(newValue)} variant="fullWidth" sx={{flexShrink:0, borderBottom:1, borderColor:'divider'}}>
                                <Tab label="Probabilities & Plot" icon={<BarChartIcon/>} iconPosition="start" />
                                <Tab label="LIME Explanation" icon={<InsightsIcon/>} iconPosition="start" disabled={!selectedPredictionDetailsJson.lime_explanation || !!selectedPredictionDetailsJson.lime_explanation.error} />
                            </Tabs>

                            {/* Probabilities Tab Panel */}
                            <Box role="tabpanel" hidden={activeContentTab !== 0} sx={{pt:2, flexGrow:1, overflowY:'auto', display: activeContentTab === 0 ? 'block' : 'none'}}>
                                {activeContentTab === 0 && (
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} lg={5}>
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
                                        <Grid item xs={12} md={7} sx={{ position: 'relative', minHeight: 300  }}>
                                            <Typography variant="subtitle1" gutterBottom>Probability Distribution Plot</Typography>
                                            {isLoadingProbPlot ? <CircularProgress/> : probabilityPlotUrl ? (
                                                <PlotViewer
                                                    artifactUrl={probabilityPlotUrl}
                                                    altText="Probability Distribution Plot"
                                                    onZoom={() => openFullscreen(probabilityPlotUrl, 'Probability Distribution')}
                                                />
                                            ) : <Alert severity="info" variant="outlined" icon={false} sx={{textAlign:'center'}}>Probability plot not generated or available.</Alert>}
                                        </Grid>
                                    </Grid>
                                )}
                            </Box>

                            {/* LIME Tab Panel */}
                            <Box role="tabpanel" hidden={activeContentTab !== 1} sx={{pt:2, flexGrow:1, overflowY:'auto', display: activeContentTab === 1 ? 'block' : 'none'}}>
                                {activeContentTab === 1 && selectedPredictionDetailsJson?.lime_explanation && !selectedPredictionDetailsJson.lime_explanation.error && (
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={5}>
                                            <Typography variant="subtitle1" gutterBottom>LIME Feature Weights for "{selectedPredictionDetailsJson.lime_explanation.explained_class_name}"</Typography>
                                            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 350, flexGrow: 1, overflow: 'auto', width: 'fit-content', maxWidth: '100%' }}>
                                                <Table stickyHeader size="small">
                                                    <TableHead>
                                                        <TableRow>
                                                            <TableCell>
                                                                <TableSortLabel
                                                                    active={limeTableOrderBy === 'segmentId'}
                                                                    direction={limeTableOrderBy === 'segmentId' ? limeTableOrder : 'asc'}
                                                                    onClick={() => handleLimeTableSortRequest('segmentId')}
                                                                >
                                                                    Feature (Segment ID)
                                                                </TableSortLabel>
                                                            </TableCell>
                                                            <TableCell align="right">
                                                                <TableSortLabel
                                                                    active={limeTableOrderBy === 'weight'}
                                                                    direction={limeTableOrderBy === 'weight' ? limeTableOrder : 'asc'}
                                                                    onClick={() => handleLimeTableSortRequest('weight')}
                                                                >
                                                                    Weight
                                                                </TableSortLabel>
                                                            </TableCell>
                                                        </TableRow>
                                                    </TableHead>
                                                    <TableBody>
                                                        {sortedLimeWeights.map(fw => (
                                                            <TableRow key={fw.segmentId} hover>
                                                                <TableCell>{fw.segmentId}</TableCell>
                                                                <TableCell align="right" sx={{color: fw.weight > 0 ? 'success.main' : 'error.main'}}>
                                                                    {Math.abs(fw.weight) < 0.001 && fw.weight !== 0 ?
                                                                        fw.weight.toExponential(4) :
                                                                        fw.weight.toFixed(4)}
                                                                </TableCell>
                                                            </TableRow>
                                                        ))}
                                                        {(!selectedPredictionDetailsJson.lime_explanation.feature_weights || selectedPredictionDetailsJson.lime_explanation.feature_weights.length === 0) && (
                                                            <TableRow>
                                                                <TableCell colSpan={2} align="center">No feature weights available.</TableCell>
                                                            </TableRow>
                                                        )}
                                                    </TableBody>
                                                </Table>
                                            </TableContainer>
                                            {selectedPredictionDetailsJson.lime_explanation.num_features_from_lime_run !== undefined && (
                                                <Typography variant="caption" display="block" sx={{mt:1}}>
                                                    Showing top {selectedPredictionDetailsJson.lime_explanation.num_features_from_lime_run} influential features.
                                                </Typography>
                                            )}
                                        </Grid>
                                        <Grid item xs={12} md={7}>
                                            <Typography variant="subtitle1" gutterBottom>LIME Explanation Plot</Typography>
                                            {isLoadingLimePlot ? (
                                                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
                                                    <CircularProgress />
                                                </Box>
                                            ) : limePlotUrl ? (
                                                <PlotViewer
                                                    artifactUrl={limePlotUrl}
                                                    altText="LIME Explanation Plot"
                                                    onZoom={() => openFullscreen(limePlotUrl, 'LIME Explanation', 'url')}
                                                />
                                            ) : (
                                                <Alert severity="info" variant="outlined" icon={false} sx={{textAlign:'center', mt:2}}>
                                                    LIME plot not generated or available for this prediction.
                                                </Alert>
                                            )}
                                        </Grid>
                                    </Grid>
                                )}
                                {activeContentTab === 1 && selectedPredictionDetailsJson?.lime_explanation?.error && (
                                    <Alert severity="warning" sx={{mt:2}}>
                                        LIME explanation could not be generated: {selectedPredictionDetailsJson.lime_explanation.error}
                                    </Alert>
                                )}
                                {activeContentTab === 1 && !selectedPredictionDetailsJson?.lime_explanation && !isLoadingSelectedPredictionContent && (
                                    <Alert severity="info" sx={{mt:2}}>LIME explanation data not available.</Alert>
                                )}
                            </Box>
                        </Paper>
                    )}
                    {selectedPrediction && !selectedPredictionDetailsJson && !isLoadingSelectedPredictionContent && (
                        <Paper elevation={2} sx={{p:3, textAlign:'center'}}>
                            <Typography color="error">Could not load prediction details (JSON file not found or unparsable).</Typography>
                        </Paper>
                    )}
                    <ConfirmDialog
                        open={deletePredictionConfirm.open}
                        onClose={() => setDeletePredictionConfirm({ open: false, prediction: null })}
                        onConfirm={handleConfirmDeletePrediction}
                        title="Delete Prediction?"
                        message={`Are you sure you want to delete this prediction (Model: ${deletePredictionConfirm.prediction?.model_experiment_name || 'Unknown'})? Associated plots and LIME data will be removed.`}
                        confirmText="Delete"
                    />
                </Grid>
            </Grid>

            {imageDetails && <NewPredictionModal open={newPredictionModalOpen}
                                                 onClose={() => setNewPredictionModalOpen(false)}
                                                 imageIds={Array.of(Number(routeImageId))}
                                                 onPredictionCreated={handlePredictionCreated} />}
            <ImageFullscreenModal open={fullscreenModal.open} onClose={() => setFullscreenModal(prev => ({...prev, open:false}))} imageUrl={fullscreenModal.type === 'url' ? fullscreenModal.src : null} imageBlob={fullscreenModal.type === 'blob' ? fullscreenModal.src : null} title={fullscreenModal.title} />
        </Container>
    );
};

export default ViewImagePredictionsPage;