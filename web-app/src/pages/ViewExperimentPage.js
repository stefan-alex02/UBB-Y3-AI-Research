import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    Container, Typography, Paper, Box, CircularProgress, Alert, Breadcrumbs,
    Link as MuiLink, Tabs, Tab, Grid, List, ListItem, ListItemButton, ListItemIcon,
    ListItemText, Chip, Button, IconButton, Tooltip
} from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import ArticleIcon from '@mui/icons-material/Article';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ZoomInIcon from '@mui/icons-material/ZoomIn';

import experimentService from '../services/experimentService';
import ArtifactViewer from '../components/ArtifactViewer/ArtifactViewer';
import LoadingSpinner from '../components/LoadingSpinner';
import ImageFullscreenModal from '../components/ImageFullscreenModal';
import { API_BASE_URL } from '../config';

const getArtifactType = (filename) => {
    // ... (keep this helper function as is)
    if (!filename) return 'unknown';
    const extension = filename.split('.').pop()?.toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(extension)) return 'image';
    if (extension === 'json') return 'json';
    if (extension === 'log' || extension === 'txt') return 'log';
    if (extension === 'csv') return 'csv';
    if (extension === 'pt') return 'model';
    return 'unknown';
};

const ViewExperimentPage = () => {
    const { experimentRunId: routeExperimentRunId } = useParams(); // Renamed to avoid conflict with state
    const navigate = useNavigate();

    const [experiment, setExperiment] = useState(null);
    const [artifacts, setArtifacts] = useState([]);
    const [currentArtifactListPath, setCurrentArtifactListPath] = useState('');
    const [selectedArtifactToView, setSelectedArtifactToView] = useState(null);

    const [isLoadingExperiment, setIsLoadingExperiment] = useState(true);
    const [isLoadingArtifactList, setIsLoadingArtifactList] = useState(false);
    const [isLoadingArtifactContent, setIsLoadingArtifactContent] = useState(false);
    const [pageError, setPageError] = useState(null);

    const [activeTab, setActiveTab] = useState(0);

    const [fullscreenModalOpen, setFullscreenModalOpen] = useState(false);
    const [fullscreenModalSource, setFullscreenModalSource] = useState({ src: null, title: '' });

    const fetchExperimentDetails = useCallback(async () => {
        if (!routeExperimentRunId) return; // Guard against missing param
        setIsLoadingExperiment(true);
        setPageError(null);
        setExperiment(null); // Reset experiment on new fetch
        try {
            const data = await experimentService.getExperimentDetails(routeExperimentRunId);
            setExperiment(data); // This is snake_case from API
        } catch (err) {
            setPageError(err.response?.data?.detail || err.message || 'Failed to fetch experiment details.');
        } finally {
            setIsLoadingExperiment(false);
        }
    }, [routeExperimentRunId]);

    const fetchArtifacts = useCallback(async (subPath = '') => {
        if (!experiment || !experiment.dataset_name || !experiment.model_type || !experiment.experiment_run_id) {
            return; // Essential guard: only proceed if experiment details are fully loaded
        }
        setIsLoadingArtifactList(true);
        setSelectedArtifactToView(null);
        setPageError(null);
        try {
            const data = await experimentService.listExperimentArtifacts(
                experiment.dataset_name, // Use snake_case
                experiment.model_type,   // Use snake_case
                experiment.experiment_run_id, // Use snake_case
                subPath
            );
            setArtifacts(data);
            setCurrentArtifactListPath(subPath);
        } catch (err) {
            setPageError(err.response?.data?.detail || err.message || 'Failed to list artifacts.');
            setArtifacts([]);
        } finally {
            setIsLoadingArtifactList(false);
        }
    }, [experiment]); // Dependency: experiment object

    useEffect(() => {
        fetchExperimentDetails();
    }, [fetchExperimentDetails]);

    useEffect(() => {
        if (experiment && !isLoadingExperiment && !pageError) {
            fetchArtifacts('');
        }
    }, [experiment, isLoadingExperiment, pageError, fetchArtifacts]);

    const handleArtifactClick = async (artifactNode) => {
        if (!experiment) return; // Guard
        if (artifactNode.type === 'folder') {
            fetchArtifacts(artifactNode.path);
        } else {
            setIsLoadingArtifactContent(true);
            setSelectedArtifactToView({ name: artifactNode.name, type: 'loading', path: artifactNode.path });
            setPageError(null);
            try {
                const artifactType = getArtifactType(artifactNode.name);
                const artifactRelativePathForFetch = artifactNode.path;

                const artifactBaseUrl = `${API_BASE_URL}/python-proxy-artifacts/experiments/${experiment.dataset_name}/${experiment.model_type}/${experiment.experiment_run_id}`;

                if (artifactType === 'image') {
                    setSelectedArtifactToView({
                        name: artifactNode.name, type: artifactType,
                        url: `${artifactBaseUrl}/${artifactRelativePathForFetch}`, content: null, path: artifactNode.path,
                    });
                } else {
                    const content = await experimentService.getExperimentArtifactContent(
                        experiment.dataset_name, experiment.model_type, experiment.experiment_run_id, artifactRelativePathForFetch
                    );
                    setSelectedArtifactToView({
                        name: artifactNode.name, type: artifactType, content: content, url: null, path: artifactNode.path,
                    });
                }
            } catch (err) {
                setPageError(`Failed to load artifact ${artifactNode.name}: ${err.message}`);
                setSelectedArtifactToView({ name: artifactNode.name, type: 'error', content: err.message, path: artifactNode.path });
            } finally {
                setIsLoadingArtifactContent(false);
            }
        }
    };

    const handleBreadcrumbClick = (pathSegmentIndex) => {
        if (currentArtifactListPath === '') return;
        const segments = currentArtifactListPath.split('/').filter(Boolean);
        const newPath = segments.slice(0, pathSegmentIndex + 1).join('/');
        fetchArtifacts(newPath);
    };

    const handleFetchExecutorLog = async () => {
        if (!experiment) return; // Guard
        setActiveTab(1);
        setIsLoadingArtifactContent(true);
        setSelectedArtifactToView({ name: "Executor Log", type: 'loading' });
        setPageError(null);
        try {
            // The experiment_run_id from the experiment object is already the correct system ID
            const logFileName = `executor_run_${experiment.experiment_run_id}.log`;
            const content = await experimentService.getExperimentArtifactContent(
                experiment.dataset_name, experiment.model_type, experiment.experiment_run_id, logFileName
            );
            setSelectedArtifactToView({ name: logFileName, type: 'log', content: content, url: null, path: logFileName });
        } catch (err) {
            setPageError(`Failed to load executor log: ${err.message}`);
            setSelectedArtifactToView({ name: "Executor Log", type: 'error', content: err.message, path: 'executor_run_log.log' });
        } finally {
            setIsLoadingArtifactContent(false);
        }
    };

    const openFullscreenArtifact = (artifact) => {
        if (!artifact || !artifact.url || artifact.type !== 'image') return;
        setFullscreenModalSource({ src: artifact.url, title: artifact.name });
        setFullscreenModalOpen(true);
    };

    // Helper to format date. Assuming timestamp is in seconds.
    const formatDate = (timestampSeconds) => {
        if (!timestampSeconds) return 'N/A';
        try { return new Date(timestampSeconds * 1000).toLocaleString(); }
        catch (e) { return 'Invalid Date'; }
    };

    // Primary loading guard for the whole page content related to 'experiment'
    if (isLoadingExperiment) return <Container sx={{mt:2}}><LoadingSpinner /></Container>;
    if (pageError && !experiment) return <Container sx={{mt:2}}><Alert severity="error" onClose={() => setPageError(null)}>{pageError}</Alert></Container>;
    if (!experiment) return <Container sx={{mt:2}}><Alert severity="info">Experiment data could not be loaded or does not exist.</Alert></Container>;

    // Now 'experiment' is guaranteed to be non-null here for the rest of the JSX
    const breadcrumbSegments = currentArtifactListPath.split('/').filter(Boolean);

    return (
        <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
            <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/experiments')} sx={{ mb: 2 }}>
                Back to Experiments
            </Button>

            <Paper elevation={3} sx={{ p: {xs:2, md:3}, mb: 3 }}>
                <Typography variant="h4" gutterBottom>{experiment.name}</Typography>
                <Grid container spacing={1}>
                    <Grid item xs={12} sm={6}><Typography><strong>Run ID:</strong> <Box component="span" sx={{fontFamily:'monospace', wordBreak:'break-all'}}>{experiment.experiment_run_id}</Box></Typography></Grid>
                    <Grid item xs={12} sm={6}><Typography><strong>Status:</strong> <Chip label={experiment.status} size="small" /></Typography></Grid>
                    <Grid item xs={12} sm={6}><Typography><strong>Model:</strong> {experiment.model_type}</Typography></Grid>
                    <Grid item xs={12} sm={6}><Typography><strong>Dataset:</strong> {experiment.dataset_name}</Typography></Grid>
                    <Grid item xs={12} sm={6}><Typography><strong>Started:</strong> {formatDate(experiment.start_time)}</Typography></Grid>
                    {experiment.end_time && <Grid item xs={12} sm={6}><Typography><strong>Ended:</strong> {formatDate(experiment.end_time)}</Typography></Grid>}
                    <Grid item xs={12}><Typography><strong>Initiated by:</strong> {experiment.user_name}</Typography></Grid>
                    {experiment.model_relative_path &&
                        <Grid item xs={12}><Typography><strong>Saved Model Path (relative):</strong> {experiment.model_relative_path}</Typography></Grid>
                    }
                </Grid>
            </Paper>

            <Tabs value={activeTab} onChange={(e, newValue) => {
                setActiveTab(newValue);
                setSelectedArtifactToView(null); // Clear view when switching main tabs
                setPageError(null); // Clear old errors
                if (newValue === 0) { // Artifacts tab
                    fetchArtifacts(currentArtifactListPath || ''); // Fetch current or root
                } else if (newValue === 1) { // Log tab
                    handleFetchExecutorLog();
                }
            }} aria-label="experiment details tabs" sx={{mb: 2, borderBottom: 1, borderColor: 'divider'}}>
                <Tab label="Artifacts Browser" id="tab-artifacts" aria-controls="tabpanel-artifacts" />
                <Tab label="Executor Log" id="tab-log" aria-controls="tabpanel-log"/>
            </Tabs>

            {/* Tab Panel for Artifacts Browser */}
            <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-artifacts" aria-labelledby="tab-artifacts">
                {activeTab === 0 && (
                    <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                            <Paper elevation={1} sx={{ p: 1, minHeight: 'calc(100vh - 450px)', maxHeight: 'calc(100vh - 450px)', overflowY:'auto' }}>
                                <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 1, p:1, borderBottom: '1px solid #eee' }}>
                                    <MuiLink component="button" onClick={() => fetchArtifacts('')} sx={{cursor: 'pointer', fontWeight: currentArtifactListPath === '' ? 'bold' : 'normal'}}>
                                        Root ({experiment.experiment_run_id.substring(0,8)}...)
                                    </MuiLink>
                                    {breadcrumbSegments.map((segment, index) => (
                                        <MuiLink component="button" key={index} onClick={() => handleBreadcrumbClick(index)} sx={{cursor: 'pointer', fontWeight: index === breadcrumbSegments.length -1 ? 'bold' : 'normal'}}>
                                            {segment}
                                        </MuiLink>
                                    ))}
                                </Breadcrumbs>
                                {isLoadingArtifactList ? <LoadingSpinner /> : (
                                    <List dense>
                                        {artifacts.sort((a,b) => (a.type === b.type ? a.name.localeCompare(b.name) : (a.type === 'folder' ? -1 : 1) )).map((node) => (
                                            <ListItem key={node.path} disablePadding>
                                                <ListItemButton onClick={() => handleArtifactClick(node)} selected={selectedArtifactToView?.path === node.path}>
                                                    <ListItemIcon sx={{minWidth: '36px'}}>
                                                        {node.type === 'folder' ? <FolderIcon fontSize="small" /> : <ArticleIcon fontSize="small" />}
                                                    </ListItemIcon>
                                                    <ListItemText primary={node.name} primaryTypographyProps={{ variant: 'body2', noWrap: true, title: node.name }} />
                                                    {getArtifactType(node.name) === 'image' && (
                                                        <Tooltip title="View Fullscreen">
                                                            <IconButton size="small" edge="end" onClick={(e) => {
                                                                e.stopPropagation();
                                                                const artifactBaseUrl = `${API_BASE_URL}/python-proxy-artifacts/experiments/${experiment.dataset_name}/${experiment.model_type}/${experiment.experiment_run_id}`;
                                                                openFullscreenArtifact({name: node.name, url: `${artifactBaseUrl}/${node.path}`, type: 'image'});
                                                            }}> <ZoomInIcon fontSize="small"/> </IconButton>
                                                        </Tooltip>
                                                    )}
                                                </ListItemButton>
                                            </ListItem>
                                        ))}
                                        {artifacts.length === 0 && !isLoadingArtifactList && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No artifacts in this folder.</Typography>}
                                    </List>
                                )}
                            </Paper>
                        </Grid>
                        <Grid item xs={12} md={8}>
                            <Box sx={{ minHeight: 'calc(100vh - 450px)', maxHeight: 'calc(100vh - 450px)', overflowY:'auto',  border: '1px solid #ddd', p: selectedArtifactToView ? 0 : 2, borderRadius: 1, backgroundColor: selectedArtifactToView && selectedArtifactToView.type !== 'loading' && selectedArtifactToView.type !== 'error' ? 'transparent': 'action.hover' }}>
                                {isLoadingArtifactContent && selectedArtifactToView?.type === 'loading' && <LoadingSpinner />}
                                {selectedArtifactToView && selectedArtifactToView.type !== 'loading' && selectedArtifactToView.type !== 'error' && (
                                    <ArtifactViewer
                                        artifactName={selectedArtifactToView.name}
                                        artifactType={selectedArtifactToView.type}
                                        artifactContent={selectedArtifactToView.content}
                                        artifactUrl={selectedArtifactToView.url}
                                        title={selectedArtifactToView.name}
                                    />
                                )}
                                {selectedArtifactToView?.type === 'error' && <Alert severity="error" sx={{m:1}}>{selectedArtifactToView.content || "Error loading artifact."}</Alert>}
                                {!selectedArtifactToView && !isLoadingArtifactContent && !pageError &&(
                                    <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: '30%' }}>
                                        Select an artifact from the list to view its content.
                                    </Typography>
                                )}
                                {pageError && (!selectedArtifactToView || selectedArtifactToView.type !== 'error') && <Alert severity="warning" sx={{m:1}}>{pageError}</Alert>}
                            </Box>
                        </Grid>
                    </Grid>
                )}
            </Box>

            {/* Tab Panel for Executor Log */}
            <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-log" aria-labelledby="tab-log">
                {activeTab === 1 && (
                    <Box sx={{ minHeight: 'calc(100vh - 350px)', maxHeight: 'calc(100vh - 350px)', overflowY:'auto', border: '1px solid #ddd', p: selectedArtifactToView ? 0 : 2, borderRadius: 1, backgroundColor: selectedArtifactToView && selectedArtifactToView.type !== 'loading' && selectedArtifactToView.type !== 'error' ? 'transparent': 'action.hover' }}>
                        {isLoadingArtifactContent && selectedArtifactToView?.name?.startsWith("executor_run_") && selectedArtifactToView?.type === 'loading' && <LoadingSpinner />}

                        {selectedArtifactToView && selectedArtifactToView.name?.startsWith("executor_run_") && selectedArtifactToView.type === 'log' && (
                            <ArtifactViewer
                                artifactName={selectedArtifactToView.name}
                                artifactType={'log'} // Force type to log
                                artifactContent={selectedArtifactToView.content}
                                title={selectedArtifactToView.name}
                            />
                        )}
                        {selectedArtifactToView && selectedArtifactToView.name?.startsWith("executor_run_") && selectedArtifactToView.type === 'error' && (
                            <Alert severity="error" sx={{m:1}}>{selectedArtifactToView.content || "Error loading executor log."}</Alert>
                        )}
                        {(!selectedArtifactToView || !selectedArtifactToView.name?.startsWith("executor_run_")) && !isLoadingArtifactContent && !pageError && (
                            <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: '30%' }}>
                                Executor log will be displayed here once loaded.
                            </Typography>
                        )}
                        {pageError && <Alert severity="warning" sx={{m:1}}>{pageError}</Alert>}
                    </Box>
                )}
            </Box>

            <ImageFullscreenModal
                open={fullscreenModalOpen}
                onClose={() => setFullscreenModalOpen(false)}
                imageUrl={fullscreenModalSource.src} // Assumes src is always a URL string for this modal
                title={fullscreenModalSource.title}
            />
        </Container>
    );
};

export default ViewExperimentPage;