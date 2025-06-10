import React, {useCallback, useEffect, useState} from 'react';
import {useNavigate, useParams} from 'react-router-dom';
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Alert,
    Box,
    Breadcrumbs,
    Button,
    Chip,
    CircularProgress,
    Container,
    Grid,
    Link as MuiLink,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Paper,
    Tab,
    Tabs,
    Typography
} from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import ArticleIcon from '@mui/icons-material/Article';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DescriptionIcon from '@mui/icons-material/Description';
import AssessmentIcon from '@mui/icons-material/Assessment';
import ImageIcon from '@mui/icons-material/Image';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';


import experimentService from '../services/experimentService';
import ArtifactViewer from '../components/ArtifactViewer/ArtifactViewer'; // We'll update this for CSV sort
import LoadingSpinner from '../components/LoadingSpinner';
import ImageFullscreenModal from '../components/Modals/ImageFullscreenModal';
import JsonViewer from '../components/ArtifactViewer/JsonViewer'; // Assuming JsonViewer is separate

const getArtifactType = (filename) => {
    if (!filename) return 'unknown';
    const extension = filename.split('.').pop()?.toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(extension)) return 'image';
    if (extension === 'json') return 'json';
    if (extension === 'log' || extension === 'txt') return 'log';
    if (extension === 'csv') return 'csv';
    if (extension === 'pt') return 'model';
    return 'file';
};

const getArtifactIcon = (type) => {
    switch(type) {
        case 'folder': return <FolderIcon fontSize="small" />;
        case 'json': return <DescriptionIcon fontSize="small" sx={{color: 'info.main'}} />;
        case 'log': return <DescriptionIcon fontSize="small" sx={{color: 'text.secondary'}}/>;
        case 'csv': return <AssessmentIcon fontSize="small" sx={{color: 'success.main'}}/>;
        case 'image': return <ImageIcon fontSize="small" sx={{color: 'secondary.main'}}/>;
        case 'model': return <ArticleIcon fontSize="small" sx={{color: 'warning.main'}}/>; // Model icon
        default: return <ArticleIcon fontSize="small" />;
    }
};

const formatDate = (timestampSeconds) => {
    if (timestampSeconds === null || timestampSeconds === undefined) return 'N/A';
    // Check if it's already a Date object (e.g., from DateTimePicker state)
    if (timestampSeconds instanceof Date) return timestampSeconds.toLocaleString();
    try {
        // Assuming timestamp is in seconds, convert to milliseconds for Date constructor
        const date = new Date(Number(timestampSeconds) * 1000);
        if (isNaN(date.getTime())) return 'Invalid Date'; // Check if date is valid
        return date.toLocaleString([], { year: 'numeric', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch (e) {
        console.error("Error formatting date:", timestampSeconds, e);
        return 'Invalid Date';
    }
};


const ViewExperimentPage = () => {
    const { experimentRunId: routeExperimentRunId } = useParams();
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
    const [fullscreenModalSource, setFullscreenModalSource] = useState({ src: null, type: 'url', title: '' });

    const [csvOrder, setCsvOrder] = useState('asc');
    const [csvOrderBy, setCsvOrderBy] = useState('');

    // Refs for calculating heights
    const headerRef = React.useRef(null);
    // Refs for height calculation (keep these)
    const pageHeaderRef = React.useRef(null); // For the Button + Paper(details) + Tabs block
    // const tabsRef = React.useRef(null); // Now part of pageHeaderRef for simplicity

    const [paneHeight, setPaneHeight] = useState('500px'); // Default, will be calculated

    useEffect(() => {
        const calculatePaneHeight = () => {
            if (pageHeaderRef.current) {
                const appBarHeight = 64; // Standard MUI AppBar height
                const containerVerticalPadding = 16 + 16; // mt={2} + mb={2} on Container approx (theme.spacing(2)*2)
                                                          // Or measure precisely if fixed
                const headerActualHeight = pageHeaderRef.current.offsetHeight;
                const calculatedHeight = `calc(100vh - ${appBarHeight}px - ${containerVerticalPadding}px - ${headerActualHeight}px - 16px)`; // Extra 16px for spacing
                setPaneHeight(calculatedHeight);
            }
        };

        // Calculate on mount and resize
        calculatePaneHeight();
        window.addEventListener('resize', calculatePaneHeight);
        return () => window.removeEventListener('resize', calculatePaneHeight);
    }, [isLoadingExperiment]); // Recalculate if header content might change after loading


    const fetchExperimentDetails = useCallback(async () => {
        if (!routeExperimentRunId) return;
        setIsLoadingExperiment(true);
        setPageError(null);
        try {
            const data = await experimentService.getExperimentDetails(routeExperimentRunId);
            setExperiment(data);
        } catch (err) {
            setPageError(err.response?.data?.message || err.message || 'Failed to fetch experiment details.');
            setExperiment(null);
        } finally {
            setIsLoadingExperiment(false);
        }
    }, [routeExperimentRunId]);

    const fetchArtifacts = useCallback(async (subPath = '') => {
        if (!routeExperimentRunId) return; // Use route param directly as experiment state might not be set yet
        setIsLoadingArtifactList(true);
        setSelectedArtifactToView(null);
        setPageError(null); // Clear previous artifact errors
        try {
            const data = await experimentService.listExperimentArtifacts(routeExperimentRunId, subPath);
            setArtifacts(data);
            setCurrentArtifactListPath(subPath);
        } catch (err) {
            setPageError(err.response?.data?.message || err.message || `Failed to list artifacts for path: '${subPath || "root"}'`);
            setArtifacts([]);
        } finally {
            setIsLoadingArtifactList(false);
        }
    }, [experiment, routeExperimentRunId]); // Depends only on experimentRunId from route

    useEffect(() => { fetchExperimentDetails(); }, [fetchExperimentDetails]);
    useEffect(() => {
        if (experiment && !isLoadingExperiment && !pageError) { fetchArtifacts(''); }
    }, [experiment, isLoadingExperiment, pageError, fetchArtifacts]);

    const handleArtifactClick = async (artifactNode) => {
        // Reset CSV sort state whenever a new artifact is clicked
            setCsvOrder('asc'); // Or whatever your default is
        setCsvOrderBy('');

        if (artifactNode.type === 'folder') {
            fetchArtifacts(artifactNode.path);
        } else if (artifactNode.type === 'model') {
            setSelectedArtifactToView({ name: artifactNode.name, type: 'model', content: "Model files (.pt, .pth) are not viewable.", path: artifactNode.path, url: null });
            setIsLoadingArtifactContent(false); return;
        } else {
            setIsLoadingArtifactContent(true);
            setSelectedArtifactToView({ name: artifactNode.name, type: 'loading', path: artifactNode.path });
            setPageError(null);
            try {
                const artifactType = getArtifactType(artifactNode.name);
                const artifactRelativePathForFetch = artifactNode.path;
                const contentOrBlob = await experimentService.getExperimentArtifactContent(routeExperimentRunId, artifactRelativePathForFetch);

                if (artifactType === 'image') {
                    const imageUrl = URL.createObjectURL(contentOrBlob);
                    setSelectedArtifactToView({ name: artifactNode.name, type: artifactType, url: imageUrl, blobContent: contentOrBlob, content: null, path: artifactNode.path });
                } else {
                    setSelectedArtifactToView({ name: artifactNode.name, type: artifactType, content: contentOrBlob, url: null, path: artifactNode.path });
                }
            } catch (err) {
                const errorMsg = err.response?.data?.message || err.message || `Failed to load artifact ${artifactNode.name}`;
                setPageError(errorMsg);
                setSelectedArtifactToView({ name: artifactNode.name, type: 'error', content: errorMsg, path: artifactNode.path });
            } finally { setIsLoadingArtifactContent(false); }
        }
    };

    // Cleanup for Object URLs when selectedArtifactToView changes
    useEffect(() => {
        const currentArtifact = selectedArtifactToView;
        if (currentArtifact && currentArtifact.type === 'image' && currentArtifact.url && currentArtifact.url.startsWith('blob:')) {
            // This was an object URL, we should revoke it when it's no longer the selected one or component unmounts
            return () => {
                URL.revokeObjectURL(currentArtifact.url);
                // console.log("Revoked object URL:", currentArtifact.url);
            };
        }
    }, [selectedArtifactToView]);


    const handleBreadcrumbClick = (pathSegmentIndex) => {
        if (currentArtifactListPath === '') return;
        const segments = currentArtifactListPath.split('/').filter(Boolean);
        const newPath = segments.slice(0, pathSegmentIndex + 1).join('/');
        fetchArtifacts(newPath);
    };

    const handleFetchExecutorLog = async () => {
        if (!experiment) return; // Guard
        setActiveTab(1); // Switch to log tab
        setIsLoadingArtifactContent(true);
        setSelectedArtifactToView({ name: `executor_run_${experiment.experiment_run_id}.log`, type: 'loading' });
        setPageError(null);
        try {
            const logFileName = `executor_run_${experiment.experiment_run_id}.log`;
            const content = await experimentService.getExperimentArtifactContent(
                experiment.experiment_run_id, logFileName
            );
            setSelectedArtifactToView({ name: logFileName, type: 'log', content: content, url: null, path: logFileName });
        } catch (err) {
            const errorMsg = err.response?.data?.message || err.message || 'Failed to load executor log.';
            setPageError(errorMsg);
            setSelectedArtifactToView({ name: `executor_run_${experiment.experiment_run_id}.log`, type: 'error', content: errorMsg, path: `executor_run_${experiment.experiment_run_id}.log` });
        } finally {
            setIsLoadingArtifactContent(false);
        }
    };

    const openFullscreenArtifact = (artifactForZoom) => {
        if (!artifactForZoom) return;
        let source = null;
        let sourceType = 'url';
        let title = artifactForZoom.name;

        if (artifactForZoom.type === 'image') { // Only allow zoom for actual images
            if (artifactForZoom.url && artifactForZoom.url.startsWith('blob:')) {
                source = artifactForZoom.url;
            } else if (artifactForZoom.blobContent instanceof Blob) {
                source = artifactForZoom.blobContent;
                sourceType = 'blob';
            } else if (artifactForZoom.url) { // Direct URL from server (less likely with current flow)
                source = artifactForZoom.url;
            }
        }

        if (source) {
            setFullscreenModalSource({ src: source, type: sourceType, title: title });
            setFullscreenModalOpen(true);
        } else if (artifactForZoom.type === 'image') {
            setPageError("Cannot display image: No valid source (URL or Blob).");
        }
    };

    const handleCsvSortRequest = (property) => {
        if (csvOrderBy === property) { // Clicked on the same column
            if (csvOrder === 'asc') {
                setCsvOrder('desc');
            } else if (csvOrder === 'desc') {
                // Cycle to "no sort" for this column, effectively resetting to original (if data isn't re-fetched)
                // Or, to truly go to "original unsorted", clear csvOrderBy
                setCsvOrder('asc'); // Default for next click on a *different* column
                setCsvOrderBy('');  // Clear orderBy to signify no sort / original order
            }
        } else { // Clicked on a new column
            setCsvOrder('asc');
            setCsvOrderBy(property);
        }
    };

    if (isLoadingExperiment) return <Container sx={{display:'flex', justifyContent:'center', alignItems:'center', height:'calc(100vh - 64px)'}}><CircularProgress size={50}/><Typography sx={{ml:2}}>Loading experiment details...</Typography></Container>;
    if (pageError && !experiment) return <Container sx={{mt:2}}><Alert severity="error" onClose={() => setPageError(null)}>{pageError}</Alert></Container>;
    if (!experiment) return <Container sx={{mt:2}}><Alert severity="info">Experiment data not available or ID not found.</Alert></Container>;

    const breadcrumbSegments = currentArtifactListPath.split('/').filter(Boolean);

    return (
        <Container
            maxWidth="xl"
            sx={{
                // Overall page container that fills viewport height minus AppBar
                height: 'calc(100vh - 64px)', // Assuming AppBar is 64px
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden', // CRUCIAL: Prevent this container from scrolling
                pt: 2, pb: 2, // Padding for the container itself
            }}
        >
            <Box ref={pageHeaderRef} sx={{ flexShrink: 0, mb: 1 }}> {/* Container for all top elements */}
                <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/experiments')} sx={{ mb: 2, alignSelf: 'flex-start' }}>
                    Back to Experiments
                </Button>

                <Paper elevation={3} sx={{ p: {xs:1.5, md:2}, mb: 2 }}>
                    <Typography variant="h5" component="h1" gutterBottom>{experiment.name}</Typography> {/* Slightly smaller H for space */}
                    <Grid container spacing={1} sx={{fontSize: '0.875rem'}}> {/* Smaller base font for details */}
                        <Grid item xs={12}><Typography variant="body2"><strong>Run ID:</strong> <Box component="span" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>{experiment.experiment_run_id}</Box></Typography></Grid>
                        <Grid item xs={6} sm={4} md={2}><Typography variant="body2"><strong>Status:</strong> <Chip label={experiment.status} size="small" color={experiment.status === 'COMPLETED' ? 'success' : experiment.status === 'FAILED' ? 'error' : 'info'} /></Typography></Grid>
                        <Grid item xs={6} sm={4} md={2}><Typography variant="body2"><strong>Model Type:</strong> {experiment.model_type}</Typography></Grid>
                        <Grid item xs={6} sm={4} md={2}><Typography variant="body2"><strong>Dataset:</strong> {experiment.dataset_name}</Typography></Grid>
                        <Grid item xs={6} sm={4} md={3}><Typography variant="body2"><strong>Started:</strong> {formatDate(experiment.start_time)}</Typography></Grid>
                        {experiment.end_time && <Grid item xs={6} sm={4} md={3}><Typography variant="body2"><strong>Ended:</strong> {formatDate(experiment.end_time)}</Typography></Grid>}
                        <Grid item xs={12} sm={4} md={3}><Typography variant="body2"><strong>Initiated by:</strong> {experiment.user_name}</Typography></Grid>
                        {experiment.model_relative_path &&
                            <Grid item xs={12} md={9}><Typography variant="body2"><strong>Saved Model:</strong> {experiment.model_relative_path}</Typography></Grid>
                        }
                    </Grid>
                    {/* Sequence Config Accordion */}
                    {experiment.sequence_config && (
                        <Box sx={{ mt: 1.5, width: '100%' }}> {/* Ensure it's below and full width */}
                            <Accordion variant="outlined" elevation={0} sx={{ '&:before': {display: 'none'}, border: theme => `1px solid ${theme.palette.divider}` }}>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{minHeight: '40px', '&.Mui-expanded': {minHeight: '40px'}}}>
                                    <Typography variant="body2"><strong>Sequence Configuration</strong></Typography>
                                </AccordionSummary>
                                <AccordionDetails sx={{p:1, maxHeight: '200px', overflowY: 'auto' }}>
                                    <JsonViewer jsonData={typeof experiment.sequence_config === 'string' ? JSON.parse(experiment.sequence_config) : experiment.sequence_config} title="" />
                                </AccordionDetails>
                            </Accordion>
                        </Box>
                    )}
                </Paper>

                {/* Tabs for Artifacts and Log */}
                <Tabs ref={pageHeaderRef} value={activeTab} onChange={(e, newValue) => {
                    setActiveTab(newValue);
                    setSelectedArtifactToView(null);
                    setPageError(null);

                    setCsvOrder('asc'); // Reset sort here
                    setCsvOrderBy('');  // Reset sort here

                    if (newValue === 0) { fetchArtifacts(currentArtifactListPath || ''); }
                    else if (newValue === 1) { handleFetchExecutorLog(); }
                }} sx={{ borderBottom: 1, borderColor: 'divider', minHeight: '48px' }}>
                    <Tab label="Artifacts Browser" id="tab-artifacts" aria-controls="tabpanel-artifacts" />
                    <Tab label="Executor Log" id="tab-log" aria-controls="tabpanel-log"/>
                </Tabs>
            </Box>

            {/* Two-Pane Layout for Artifacts/Log */}
            <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden', minHeight: 0 }}>
                {/* Left Pane: Artifact List */}
                {activeTab === 0 && (
                    <Paper
                        variant="outlined"
                        sx={{
                            width: { xs: '100%', sm: 280, md: 320 }, // Responsive width
                            flexShrink: 0,
                            mr: {sm: 1.5}, // Margin only on larger screens
                            mb: {xs: 1.5, sm: 0}, // Margin bottom on small screens if stacked
                            height: {xs: 'auto', sm: paneHeight}, // Auto height on small screens, calculated on larger
                            minHeight: {xs: 200, sm: 'auto'}, // Min height on small screens
                            display: 'flex',
                            flexDirection: 'column',
                            overflow: 'hidden', // This Paper controls its children's overflow
                        }}
                    >
                        <Breadcrumbs aria-label="breadcrumb" sx={{ p:1, borderBottom: '1px solid #eee', flexShrink:0, overflowX: 'auto', whiteSpace:'nowrap' }}>
                                    <MuiLink component="button" onClick={() => fetchArtifacts('')} sx={{cursor: 'pointer', fontWeight: currentArtifactListPath === '' ? 'bold' : 'normal'}}>
                                        Root ({experiment.experiment_run_id.substring(0,8)}...)
                                    </MuiLink>
                                    {breadcrumbSegments.map((segment, index) => (
                                        <MuiLink component="button" key={index} onClick={() => handleBreadcrumbClick(index)} sx={{cursor: 'pointer', fontWeight: index === breadcrumbSegments.length -1 ? 'bold' : 'normal'}}>
                                            {segment}
                                        </MuiLink>
                                    ))}
                        </Breadcrumbs>
                        <Box sx={{flexGrow: 1, overflowY: 'auto', overflowX: 'auto' }}>
                            {isLoadingArtifactList ? <Box sx={{p:2, textAlign:'center'}}><CircularProgress size={24}/></Box> : (
                                <List dense sx={{py:0.5}}>
                                    {artifacts.map((node) => ( /* Sort is now in Python service */
                                        <ListItem key={node.path} disablePadding sx={{ '&:hover': { bgcolor: 'action.hover' }}}>
                                            <ListItemButton dense onClick={() => handleArtifactClick(node)} selected={selectedArtifactToView?.path === node.path} sx={{pl:1}}>
                                                <ListItemIcon sx={{minWidth: '30px'}}>{getArtifactIcon(node.type)}</ListItemIcon>
                                                <ListItemText primary={node.name} primaryTypographyProps={{ variant: 'body2', noWrap: true, title: node.name, fontSize: '0.8rem' }} />
                                            </ListItemButton>
                                        </ListItem>
                                    ))}
                                    {artifacts.length === 0 && !isLoadingArtifactList && <Typography sx={{p:2, textAlign:'center', color:'text.secondary'}} variant="body2">No artifacts in this folder.</Typography>}
                                </List>
                            )}
                        </Box>
                    </Paper>
                )}

                {/* Right Pane: Content Viewer */}
                <Paper
                    variant="outlined"
                    sx={{
                        flexGrow: 1,
                        height: paneHeight, // Use calculated height
                        display: 'flex',
                        flexDirection: 'column',
                        overflow: 'hidden', // This Paper also controls children's overflow
                        minWidth: 0, // Important for flex item to shrink properly
                    }}
                >
                    {activeTab === 0 && ( // Artifact Browser Content
                        isLoadingArtifactContent && selectedArtifactToView?.type === 'loading' ? <Box sx={{m:'auto'}}><LoadingSpinner/></Box> :
                            selectedArtifactToView && selectedArtifactToView.type !== 'loading' && selectedArtifactToView.type !== 'error' ? (
                                <ArtifactViewer
                                    artifactName={selectedArtifactToView.name} artifactType={selectedArtifactToView.type}
                                    artifactContent={selectedArtifactToView.content} artifactUrl={selectedArtifactToView.url}
                                    title={selectedArtifactToView.name}
                                    onImageZoom={() => openFullscreenArtifact(selectedArtifactToView)}
                                    csvOrder={csvOrder} csvOrderBy={csvOrderBy} onCsvSortRequest={handleCsvSortRequest}
                                    isSortable={selectedArtifactToView.type === 'csv'}
                                />
                            ) : selectedArtifactToView?.type === 'error' ? <Alert severity="error" sx={{m:1}}>{selectedArtifactToView.content}</Alert>
                                : <Box sx={{m:'auto', textAlign:'center'}}><Typography color="text.secondary">Select an artifact to view.</Typography></Box>
                    )}
                    {activeTab === 1 && ( // Executor Log Content
                        isLoadingArtifactContent && selectedArtifactToView?.name?.startsWith("executor_run_") ? <Box sx={{m:'auto'}}><LoadingSpinner/></Box> :
                            selectedArtifactToView && selectedArtifactToView.name?.startsWith("executor_run_") && selectedArtifactToView.type === 'log' ? (
                                <ArtifactViewer artifactName={selectedArtifactToView.name} artifactType={'log'}
                                                artifactContent={selectedArtifactToView.content} title={selectedArtifactToView.name} />
                            ) : selectedArtifactToView?.type === 'error' && selectedArtifactToView.name?.startsWith("executor_run_") ? <Alert severity="error" sx={{m:1}}>{selectedArtifactToView.content}</Alert>
                                : <Box sx={{m:'auto', textAlign:'center'}}><Typography color="text.secondary" >Executor log will be displayed here.</Typography></Box>
                    )}
                    {/* General page error for artifact fetching issues */}
                    {pageError && (!selectedArtifactToView || selectedArtifactToView.type !== 'error') &&
                        !isLoadingArtifactContent && !isLoadingArtifactList && /* Only show if not actively loading */
                        <Alert severity="warning" sx={{m:1, position: 'absolute', bottom: 8, left: 'calc(320px + 24px)', right: 8, zIndex:10}}> {/* Adjust left based on left pane width + margin */}
                            {pageError}
                        </Alert>
                    }
                </Paper>
            </Box>

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

export default ViewExperimentPage;