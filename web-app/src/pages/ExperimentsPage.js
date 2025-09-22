import React, {useCallback, useEffect, useRef, useState} from 'react';
import {
    Alert,
    Box,
    Button,
    CircularProgress,
    Collapse,
    Container,
    FormControl,
    FormControlLabel,
    Grid,
    IconButton,
    InputLabel,
    MenuItem,
    Pagination,
    Paper,
    Select,
    Switch,
    TextField,
    Tooltip,
    Typography
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import FilterListIcon from '@mui/icons-material/FilterList';
import RefreshIcon from '@mui/icons-material/Refresh';
import {useNavigate} from 'react-router-dom';
import {AdapterDateFns} from '@mui/x-date-pickers/AdapterDateFns';
import {LocalizationProvider} from '@mui/x-date-pickers/LocalizationProvider';
import {DateTimePicker} from '@mui/x-date-pickers/DateTimePicker';

import ExperimentGrid from '../components/ExperimentGrid/ExperimentGrid';
import experimentService from '../services/experimentService';
import useAuth from '../hooks/useAuth';
import LoadingSpinner from '../components/LoadingSpinner';
import ConfirmDialog from '../components/ConfirmDialog';
import {DATASET_NAMES, MODEL_TYPES} from './experimentConfig';

const ExperimentsPage = () => {
    const navigate = useNavigate();
    const { user } = useAuth();

    const [experiments, setExperiments] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const initialFilters = {
        nameContains: '',
        modelType: '',
        datasetName: '',
        status: '',
        hasModelSaved: null,
        startedAfter: null,
        finishedBefore: null,
    };
    const [filters, setFilters] = useState(initialFilters);
    const [pagination, setPagination] = useState({
        page: 0, size: 12, totalPages: 0, totalElements: 0,
    });
    const [showFilters, setShowFilters] = useState(false);
    const [deleteConfirm, setDeleteConfirm] = useState({ open: false, experimentId: null, experimentName: '' });

    const [autoUpdate, setAutoUpdate] = useState(true);
    const [isWsConnected, setIsWsConnected] = useState(false);
    const webSocketRef = useRef(null);
    const WS_URL = `ws://${window.location.hostname}:8080/ws/experiment-status`;


    const fetchExperiments = useCallback(async (isManualRefresh = false) => {
        if (!user || user.role !== 'METEOROLOGIST') {
            setIsLoading(false); setExperiments([]); return;
        }
        setIsLoading(true); setError(null);
        try {
            const pageable = { page: pagination.page, size: pagination.size, sortBy: 'startTime', sortDir: 'DESC' };
            const activeFilters = { ...filters };
            if (activeFilters.startedAfter) activeFilters.startedAfter = new Date(activeFilters.startedAfter).toISOString();
            if (activeFilters.finishedBefore) activeFilters.finishedBefore = new Date(activeFilters.finishedBefore).toISOString();

            if (activeFilters.hasModelSaved === null || activeFilters.hasModelSaved === "any") {
                delete activeFilters.hasModelSaved;
            }

            const data = await experimentService.getExperiments(activeFilters, pageable);
            setExperiments(data.content || []);
            setPagination(prev => ({ ...prev, totalPages: data.total_pages, totalElements: data.total_elements }));
        } catch (err) {
            setError(err.response?.data?.message || err.message || 'Failed to fetch experiments.');
            setExperiments([]); setPagination(prev => ({ ...prev, totalPages: 0, totalElements: 0 }));
        } finally { setIsLoading(false); }
    }, [user, filters, pagination.page, pagination.size]);

    useEffect(() => { fetchExperiments(); }, [fetchExperiments]);


    const connectWebSocket = useCallback(() => {
        if (!user || user.role !== 'METEOROLOGIST' || (webSocketRef.current && webSocketRef.current.readyState === WebSocket.OPEN)) {
            return;
        }

        console.log('Attempting to connect WebSocket...');
        webSocketRef.current = new WebSocket(WS_URL);

        webSocketRef.current.onopen = () => {
            console.log('WebSocket connected for experiment status updates');
            setIsWsConnected(true);
        };

        webSocketRef.current.onmessage = (event) => {
            try {
                const updatedExperimentData = JSON.parse(event.data);
                console.log('WebSocket message received (raw):', updatedExperimentData);

                setExperiments(prevExperiments =>
                    prevExperiments.map(exp =>
                        exp.experiment_run_id === updatedExperimentData.experiment_run_id
                            ? { ...exp, ...updatedExperimentData }
                            : exp
                    )
                );
            } catch (e) {
                console.error('Error processing WebSocket message:', e, 'Data:', event.data);
            }
        };

        webSocketRef.current.onclose = (event) => {
            console.log('WebSocket disconnected:', event.reason, `Code: ${event.code}`);
            setIsWsConnected(false);
        };

        webSocketRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            setIsWsConnected(false);
        };
    }, [user, WS_URL]);

    const disconnectWebSocket = useCallback(() => {
        if (webSocketRef.current) {
            webSocketRef.current.close();
            webSocketRef.current = null;
            setIsWsConnected(false);
            console.log('WebSocket explicitly disconnected.');
        }
    }, []);

    useEffect(() => {
        if (autoUpdate) {
            connectWebSocket();
        } else {
            disconnectWebSocket();
        }
        return () => {
            disconnectWebSocket();
        };
    }, [autoUpdate, connectWebSocket, disconnectWebSocket]);

    const handleFilterChange = (event) => {
        const { name, value, type, checked } = event.target;
        setFilters(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? (checked ? true : (name === 'hasModelSaved' ? false : checked) ) : value
        }));
        setPagination(prev => ({ ...prev, page: 0 }));
    };

    const handleHasModelSavedChange = (event) => {
        const value = event.target.value;
        setFilters(prev => ({
            ...prev,
            hasModelSaved: value === "any" ? null : (value === "true")
        }));
        setPagination(prev => ({ ...prev, page: 0 }));
    };

    const handleDateChange = (name, date) => {
        setFilters(prev => ({ ...prev, [name]: date }));
        setPagination(prev => ({ ...prev, page: 0 }));
    };

    const handleClearFilters = () => {
        setFilters({ modelType: '', datasetName: '', status: '', startedAfter: null, finishedBefore: null });
        setPagination(prev => ({ ...prev, page: 0 }));
    };

    const handlePageChange = (event, value) => {
        setPagination(prev => ({ ...prev, page: value - 1 }));
    };

    const openDeleteDialog = (id, name) => {
        const displayName = name || 'Unnamed Experiment';
        console.log(`PAGE: openDeleteDialog - Setting state to open for ID: ${id}, Name: ${displayName}`);
        setDeleteConfirm({ open: true, experimentId: id, experimentName: displayName });
    };

    const handleCloseDeleteDialog = () => {
        console.log("PAGE: handleCloseDeleteDialog - Setting state to close");
        setDeleteConfirm({ open: false, experimentId: null, experimentName: '' });
    };

    const handleConfirmDeletion = async () => {
        const idToDelete = deleteConfirm.experimentId;
        const nameToDelete = deleteConfirm.experimentName;
        console.log(`PAGE: handleConfirmDeletion - Confirming delete for ID: ${idToDelete}, Name: ${nameToDelete}`);

        if (!idToDelete) {
            console.warn("PAGE: handleConfirmDeletion called but no experimentId in state.");
            handleCloseDeleteDialog();
            return;
        }

        handleCloseDeleteDialog();

        setError(null);
        try {
            await experimentService.deleteExperiment(idToDelete);
            console.log(`PAGE: Deletion API call successful for ${idToDelete}`);
            fetchExperiments();
        } catch (err) {
            console.error(`PAGE: Deletion API call failed for ${idToDelete}`, err);
            setError(err.response?.data?.message || `Failed to delete experiment "${nameToDelete}".`);
        }
    };


    if (!user) return <LoadingSpinner />;
    if (user.role !== 'METEOROLOGIST') {
        return (
            <Container sx={{mt: 4}}>
                <Alert severity="warning">Access Denied. This page is for meteorologists only.</Alert>
            </Container>
        );
    }

    return (
        <LocalizationProvider dateAdapter={AdapterDateFns}>
            <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h4" component="h1">Experiments Dashboard</Typography>
                    <Box sx={{display: 'flex', alignItems: 'center'}}>
                        <Tooltip title={autoUpdate ? "Disable real-time status updates" : "Enable real-time status updates"}>
                            <FormControlLabel
                                control={<Switch checked={autoUpdate} onChange={(e) => setAutoUpdate(e.target.checked)} />}
                                labelPlacement="start"
                                label={autoUpdate ? (isWsConnected ? "Auto-Update: ON" : "Auto-Update: Connecting...") : "Auto-Update: OFF"}
                                sx={{mr:1}}
                            />
                        </Tooltip>
                        <Tooltip title="Manually refresh the experiment list">
                          <span>
                              <IconButton onClick={() => fetchExperiments(true)} disabled={isLoading} color="primary">
                              <RefreshIcon />
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Button variant="outlined" startIcon={<FilterListIcon />} onClick={() => setShowFilters(!showFilters)} sx={{ ml: 2, mr: 2 }}>
                            {showFilters ? 'Hide Filters' : 'Show Filters'}
                        </Button>
                        <Button color="secondary" variant="contained" startIcon={<AddIcon />} onClick={() => navigate('/experiments/create')}>New Experiment</Button>
                    </Box>
                </Box>

                <Collapse in={showFilters}>
                    <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
                        <Grid container spacing={2} alignItems="center">
                            <Grid item xs={12} sm={6} md={4} lg={3}>
                                <TextField fullWidth label="Experiment Name" name="nameContains" value={filters.nameContains} onChange={handleFilterChange} variant="outlined" size="small"/>
                            </Grid>
                            <Grid item xs={12} sm={6} md={4} lg={3}>
                                <FormControl fullWidth variant="outlined" size="small" sx={{ minWidth: '130px' }}>
                                    <InputLabel>Model Type</InputLabel>
                                    <Select name="modelType" value={filters.modelType} label="Model Type" onChange={handleFilterChange}>
                                        <MenuItem value=""><em>Any</em></MenuItem>
                                        {MODEL_TYPES.map(mt => <MenuItem key={mt.value} value={mt.value}>{mt.label}</MenuItem>)}
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12} sm={6} md={4} lg={3}>
                                <FormControl fullWidth variant="outlined" size="small" sx={{ minWidth: '130px' }}>
                                    <InputLabel>Dataset Name</InputLabel>
                                    <Select name="datasetName" value={filters.datasetName} label="Dataset Name" onChange={handleFilterChange}>
                                        <MenuItem value=""><em>Any</em></MenuItem>
                                        {DATASET_NAMES.map(dn => <MenuItem key={dn} value={dn}>{dn}</MenuItem>)}
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12} sm={6} md={4} lg={3}>
                                <FormControl fullWidth variant="outlined" size="small" sx={{ minWidth: '130px' }}>
                                    <InputLabel>Status</InputLabel>
                                    <Select name="status" value={filters.status} label="Status" onChange={handleFilterChange}> <MenuItem value=""><em>Any</em></MenuItem>
                                        <MenuItem value="PENDING">Pending</MenuItem>
                                        <MenuItem value="RUNNING">Running</MenuItem>
                                        <MenuItem value="COMPLETED">Completed</MenuItem>
                                        <MenuItem value="FAILED">Failed</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12} sm={6} md={4} lg={3}>
                                <FormControl fullWidth variant="outlined" size="small">
                                    <InputLabel>Model Saved?</InputLabel>
                                    <Select name="hasModelSaved" value={filters.hasModelSaved === null ? "any" : String(filters.hasModelSaved)} label="Model Saved?" onChange={handleHasModelSavedChange}>
                                        <MenuItem value="any"><em>Any</em></MenuItem>
                                        <MenuItem value="true">Yes</MenuItem>
                                        <MenuItem value="false">No</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12} sm={6} md={4} lg={2.5}><DateTimePicker label="Started After" value={filters.startedAfter} onChange={(d) => handleDateChange('startedAfter', d)} slotProps={{ textField: { size: 'small', fullWidth: true } }} /></Grid>
                            <Grid item xs={12} sm={6} md={4} lg={2.5}><DateTimePicker label="Finished Before" value={filters.finishedBefore} onChange={(d) => handleDateChange('finishedBefore', d)} slotProps={{ textField: { size: 'small', fullWidth: true } }} /></Grid>
                            <Grid item xs={12} md={12} lg={1} sx={{display:'flex', justifyContent: {xs:'flex-start', lg:'flex-end'}, pt: {xs:1, lg:0} }}>
                                <Button onClick={handleClearFilters} variant="text" size="medium">Clear All</Button>
                            </Grid>
                        </Grid>
                    </Paper>
                </Collapse>

                {isLoading && <Box sx={{display: 'flex', justifyContent: 'center', my: 5}}><CircularProgress /></Box>}
                {!isLoading && error && <Alert severity="error" sx={{my: 2}}>{error}</Alert>}
                {!isLoading && !error && (
                    <>
                        <Typography variant="body2" color="textSecondary" sx={{mb:1}}>
                            Showing {experiments.length} of {pagination.totalElements} experiments.
                        </Typography>
                        {experiments.length > 0 ? (
                            <ExperimentGrid experiments={experiments} onDelete={openDeleteDialog} />
                        ) : (
                            <Paper sx={{p:3, textAlign:'center', mt:3}}><Typography>No experiments found matching your criteria.</Typography></Paper>
                        )}
                        {experiments.length > 0 && pagination.totalPages > 1 && (
                            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb:2 }}>
                                <Pagination
                                    count={pagination.totalPages}
                                    page={pagination.page + 1}
                                    onChange={handlePageChange}
                                    color="primary"
                                    showFirstButton
                                    showLastButton
                                />
                            </Box>
                        )}
                    </>
                )}
            </Container>
            <ConfirmDialog
                open={deleteConfirm.open}
                onClose={handleCloseDeleteDialog}
                onConfirm={handleConfirmDeletion}
                title="Delete Experiment?"
                message={`Are you sure you want to delete the experiment "${deleteConfirm.experimentName}" (ID: ${deleteConfirm.experimentId || 'N/A'})? This action cannot be undone and will also attempt to delete associated artifacts.`}
                confirmText="Delete"
                cancelText="Cancel"
            />
        </LocalizationProvider>
    );
};

export default ExperimentsPage;