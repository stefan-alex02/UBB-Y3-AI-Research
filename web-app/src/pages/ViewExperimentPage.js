import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
    Container,
    Typography,
    Paper,
    Box,
    CircularProgress,
    Alert,
    Breadcrumbs,
    Link as MuiLink,
    Tabs,
    Tab,
    Grid,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Chip,
    Button
} from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import ArticleIcon from '@mui/icons-material/Article'; // For files
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import experimentService from '../services/experimentService';
import ArtifactViewer from '../components/ArtifactViewer/ArtifactViewer';
import LoadingSpinner from '../components/LoadingSpinner';
import { API_BASE_URL } from '../config'; // For constructing direct artifact URLs if needed

// Helper to determine artifact type from filename
const getArtifactType = (filename) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(extension)) return 'image';
    if (extension === 'json') return 'json';
    if (extension === 'log' || extension === 'txt') return 'log';
    if (extension === 'csv') return 'csv';
    // Add more types like 'pt' for model files (though not directly viewable)
    if (extension === 'pt') return 'model';
    return 'unknown';
};


const ViewExperimentPage = () => {
    const { experimentRunId } = useParams();
    const navigate = useNavigate();
    const [experiment, setExperiment] = useState(null);
    const [artifacts, setArtifacts] = useState([]); // List of ArtifactNode
    const [currentArtifactPath, setCurrentArtifactPath] = useState(''); // Relative to experiment root
    const [selectedArtifact, setSelectedArtifact] = useState(null); // { name, type, content, url }
    const [isLoading, setIsLoading] = useState(true);
    const [loadingArtifact, setLoadingArtifact] = useState(false);
    const [error, setError] = useState(null);

    const [activeTab, setActiveTab] = useState(0); // For main sections like Summary, Artifacts, Log

    const fetchExperimentDetails = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await experimentService.getExperimentDetails(experimentRunId);
            setExperiment(data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to fetch experiment details.');
        } finally {
            setIsLoading(false);
        }
    }, [experimentRunId]);

    const fetchArtifacts = useCallback(async (subPath = '') => {
        if (!experiment) return;
        setLoadingArtifact(true); // For artifact list loading
        setSelectedArtifact(null); // Clear selected artifact when path changes
        try {
            const data = await experimentService.listExperimentArtifacts(
                experiment.datasetName,
                experiment.modelType,
                experiment.experimentRunId,
                subPath
            );
            setArtifacts(data);
            setCurrentArtifactPath(subPath);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to list artifacts.');
            setArtifacts([]);
        } finally {
            setLoadingArtifact(false);
        }
    }, [experiment]);

    useEffect(() => {
        fetchExperimentDetails();
    }, [fetchExperimentDetails]);

    useEffect(() => {
        if (experiment) {
            fetchArtifacts(); // Fetch root artifacts initially
        }
    }, [experiment, fetchArtifacts]); // Re-fetch artifacts if experiment details load

    const handleArtifactClick = async (artifactNode) => {
        if (artifactNode.type === 'folder') {
            fetchArtifacts(artifactNode.path); // Path is already relative to exp root
        } else {
            setLoadingArtifact(true);
            setSelectedArtifact({ name: artifactNode.name, type: 'loading', content: null });
            try {
                const type = getArtifactType(artifactNode.name);
                const artifactBasePath = `${API_BASE_URL}/python-proxy-artifacts/experiments/${experiment.datasetName}/${experiment.modelType}/${experiment.experimentRunId}`;

                if (type === 'image') {
                    setSelectedArtifact({
                        name: artifactNode.name,
                        type: type,
                        url: `${artifactBasePath}/${artifactNode.path}`, // Full URL to image
                        content: null,
                    });
                } else {
                    const content = await experimentService.getExperimentArtifactContent(
                        experiment.datasetName,
                        experiment.modelType,
                        experiment.experimentRunId,
                        artifactNode.path // artifactNode.path is relative path like 'method_0/results.json'
                    );
                    setSelectedArtifact({ name: artifactNode.name, type: type, content: content, url: null });
                }
            } catch (err) {
                setError(`Failed to load artifact ${artifactNode.name}: ${err.message}`);
                setSelectedArtifact({ name: artifactNode.name, type: 'error', content: err.message });
            } finally {
                setLoadingArtifact(false);
            }
        }
    };

    const handleBreadcrumbClick = (pathSegmentIndex) => {
        const segments = currentArtifactPath.split('/').filter(Boolean);
        const newPath = segments.slice(0, pathSegmentIndex + 1).join('/');
        fetchArtifacts(newPath);
    };

    const getExecutorLogContent = async () => {
        setLoadingArtifact(true);
        setSelectedArtifact({ name: "Executor Log", type: 'loading', content: null });
        try {
            const logFileName = `executor_run_${experimentRunId}.log`;
            const content = await experimentService.getExperimentArtifactContent(
                experiment.datasetName,
                experiment.modelType,
                experiment.experimentRunId,
                logFileName // Log is at the root of experimentRunId folder
            );
            setSelectedArtifact({ name: logFileName, type: 'log', content: content, url: null });
        } catch (err) {
            setError(`Failed to load executor log: ${err.message}`);
            setSelectedArtifact({ name: "Executor Log", type: 'error', content: err.message });
        } finally {
            setLoadingArtifact(false);
        }
    }


    if (isLoading) return <LoadingSpinner />;
    if (error && !experiment) return <Alert severity="error">{error}</Alert>; // Show main error if experiment fails to load

    const breadcrumbSegments = currentArtifactPath.split('/').filter(Boolean);

    return (
        <Container maxWidth="xl" sx={{ mt: 2 }}>
            <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/experiments')} sx={{ mb: 2 }}>
                Back to Experiments
            </Button>
            {experiment && (
                <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h4" gutterBottom>{experiment.name}</Typography>
                    <Grid container spacing={1}>
                        <Grid item xs={12} sm={6}><Typography><strong>Run ID:</strong> {experiment.experimentRunId}</Typography></Grid>
                        <Grid item xs={12} sm={6}><Typography><strong>Status:</strong> <Chip label={experiment.status} size="small" color={selectedArtifact?.name === "Executor Log" ? "primary" : "default"} /></Typography></Grid>
                        <Grid item xs={12} sm={6}><Typography><strong>Model:</strong> {experiment.modelType}</Typography></Grid>
                        <Grid item xs={12} sm={6}><Typography><strong>Dataset:</strong> {experiment.datasetName}</Typography></Grid>
                        <Grid item xs={12} sm={6}><Typography><strong>Started:</strong> {new Date(experiment.startTime).toLocaleString()}</Typography></Grid>
                        {experiment.endTime && <Grid item xs={12} sm={6}><Typography><strong>Ended:</strong> {new Date(experiment.endTime).toLocaleString()}</Typography></Grid>}
                        <Grid item xs={12}><Typography><strong>Initiated by:</strong> {experiment.userName}</Typography></Grid>
                        {experiment.modelRelativePath &&
                            <Grid item xs={12}><Typography><strong>Saved Model:</strong> {experiment.modelRelativePath}</Typography></Grid>
                        }
                    </Grid>
                </Paper>
            )}

            <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} aria-label="experiment details tabs" sx={{mb: 2}}>
                <Tab label="Artifacts Browser" />
                <Tab label="Executor Log" onClick={getExecutorLogContent} />
                {/* You could add more tabs for a structured summary if JSON results are simple enough */}
            </Tabs>

            {activeTab === 0 && experiment && (
                <Grid container spacing={2}>
                    <Grid item xs={12} md={4}> {/* Artifact List */}
                        <Paper elevation={1} sx={{ p: 1, minHeight: 'calc(100vh - 300px)', maxHeight: 'calc(100vh - 300px)', overflowY:'auto' }}>
                            <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 1, p:1 }}>
                                <MuiLink component="button" onClick={() => fetchArtifacts('')} sx={{cursor: 'pointer'}}>
                                    {experiment.experimentRunId.substring(0,8)}... {/* Shortened ID */}
                                </MuiLink>
                                {breadcrumbSegments.map((segment, index) => (
                                    <MuiLink component="button" key={index} onClick={() => handleBreadcrumbClick(index)} sx={{cursor: 'pointer'}}>
                                        {segment}
                                    </MuiLink>
                                ))}
                            </Breadcrumbs>
                            {loadingArtifact && artifacts.length === 0 ? <LoadingSpinner /> : (
                                <List dense>
                                    {artifacts.sort((a,b) => a.type.localeCompare(b.type) || a.name.localeCompare(b.name)).map((node) => (
                                        <ListItem key={node.path} disablePadding>
                                            <ListItemButton onClick={() => handleArtifactClick(node)} selected={selectedArtifact?.name === node.name && currentArtifactPath === node.path.substring(0, node.path.lastIndexOf('/') > 0 ? node.path.lastIndexOf('/') : 0 )}>
                                                <ListItemIcon sx={{minWidth: '32px'}}>
                                                    {node.type === 'folder' ? <FolderIcon fontSize="small" /> : <ArticleIcon fontSize="small" />}
                                                </ListItemIcon>
                                                <ListItemText primary={node.name} primaryTypographyProps={{ variant: 'body2', noWrap: true }} />
                                            </ListItemButton>
                                        </ListItem>
                                    ))}
                                    {artifacts.length === 0 && !loadingArtifact && <Typography sx={{p:2, textAlign:'center'}} variant="body2">No artifacts in this folder.</Typography>}
                                </List>
                            )}
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={8}> {/* Artifact Viewer */}
                        <Box sx={{ minHeight: 'calc(100vh - 300px)', maxHeight: 'calc(100vh - 300px)', overflowY:'auto',  border: '1px dashed grey', p: selectedArtifact ? 0 : 2, borderRadius: 1 }}>
                            {loadingArtifact && selectedArtifact?.type === 'loading' && <LoadingSpinner />}
                            {selectedArtifact && selectedArtifact.type !== 'loading' && (
                                <ArtifactViewer
                                    artifactName={selectedArtifact.name}
                                    artifactType={selectedArtifact.type}
                                    artifactContent={selectedArtifact.content}
                                    artifactUrl={selectedArtifact.url}
                                />
                            )}
                            {!selectedArtifact && !loadingArtifact && (
                                <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: '30%' }}>
                                    Select an artifact from the list to view its content.
                                </Typography>
                            )}
                            {error && !loadingArtifact && <Alert severity="error" sx={{m:1}}>{error}</Alert>}
                        </Box>
                    </Grid>
                </Grid>
            )}
            {activeTab === 1 && ( // Executor Log Viewer
                <Grid container>
                    <Grid item xs={12}>
                        <Box sx={{ minHeight: 'calc(100vh - 300px)', maxHeight: 'calc(100vh - 300px)', overflowY:'auto',  border: '1px dashed grey', p: selectedArtifact ? 0 : 2, borderRadius: 1 }}>
                            {loadingArtifact && selectedArtifact?.name === "Executor Log" && selectedArtifact?.type === 'loading' && <LoadingSpinner />}
                            {selectedArtifact && selectedArtifact.name === "Executor Log" && selectedArtifact.type !== 'loading' && (
                                <ArtifactViewer
                                    artifactName={selectedArtifact.name}
                                    artifactType={selectedArtifact.type}
                                    artifactContent={selectedArtifact.content}
                                />
                            )}
                            {!selectedArtifact && !loadingArtifact && (
                                <Typography sx={{ textAlign: 'center', color: 'text.secondary', mt: '30%' }}>
                                    Click the "Executor Log" tab again or ensure log is available.
                                </Typography>
                            )}
                            {error && !loadingArtifact && <Alert severity="error" sx={{m:1}}>{error}</Alert>}
                        </Box>
                    </Grid>
                </Grid>
            )}
        </Container>
    );
};

export default ViewExperimentPage;