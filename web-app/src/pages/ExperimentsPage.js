// src/pages/ExperimentsPage.js
import React, {useCallback, useEffect, useRef, useState} from 'react';
import {
    Alert,
    Box,
    Button,
    CircularProgress,
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
import FilterListIcon from '@mui/icons-material/FilterList'; // For filter section toggle
import {useNavigate} from 'react-router-dom';
import {AdapterDateFns} from '@mui/x-date-pickers/AdapterDateFns'; // Or AdapterDayjs
import {LocalizationProvider} from '@mui/x-date-pickers/LocalizationProvider';
import {DateTimePicker} from '@mui/x-date-pickers/DateTimePicker';

import ExperimentGrid from '../components/ExperimentGrid/ExperimentGrid';
import experimentService from '../services/experimentService';
import useAuth from '../hooks/useAuth';
import LoadingSpinner from '../components/LoadingSpinner';
import ConfirmDialog from '../components/ConfirmDialog'; // Assuming this is in components/
import RefreshIcon from '@mui/icons-material/Refresh';


const ExperimentsPage = () => {
    const navigate = useNavigate();
    const { user } = useAuth();

    const [experiments, setExperiments] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filters, setFilters] = useState({
        modelType: '',
        datasetName: '',
        status: '',
        startedAfter: null,
        finishedBefore: null,
    });
    const [pagination, setPagination] = useState({
        page: 0, // 0-indexed for Spring Data Pageable
        size: 9,
        totalPages: 0,
        totalElements: 0,
    });
    const [showFilters, setShowFilters] = useState(false); // To toggle filter section visibility

    const [deleteConfirm, setDeleteConfirm] = useState({
        open: false,
        experimentId: null,
        experimentName: '',
    });

    const [autoUpdate, setAutoUpdate] = useState(true); // State for the auto-update toggle
    const [isWsConnected, setIsWsConnected] = useState(false);
    const webSocketRef = useRef(null); // To hold the WebSocket instance

    const WS_URL = `ws://${window.location.hostname}:8080/ws/experiment-status`; // Adjust port if different


    const connectWebSocket = useCallback(() => {
        if (!user || user.role !== 'METEOROLOGIST' || (webSocketRef.current && webSocketRef.current.readyState === WebSocket.OPEN)) {
            return; // Don't connect if not authorized, or already connected
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
                // Ensure keys from WebSocket match what the frontend expects (snake_case vs camelCase)
                // If Java sends snake_case (default for DTOs with global SNAKE_CASE strategy):
                console.log('WebSocket message received (raw):', updatedExperimentData);

                setExperiments(prevExperiments =>
                    prevExperiments.map(exp =>
                        exp.experiment_run_id === updatedExperimentData.experiment_run_id
                            ? { ...exp, ...updatedExperimentData } // Merge updates
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
            // Optional: Implement reconnection logic if autoUpdate is still true
            // if (autoUpdate && !event.wasClean) { setTimeout(connectWebSocket, 5000); }
        };

        webSocketRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            setIsWsConnected(false);
            // Maybe set autoUpdate to false on persistent errors
        };
    }, [user, WS_URL]); // Removed autoUpdate from here to control connection manually

    const disconnectWebSocket = useCallback(() => {
        if (webSocketRef.current) {
            webSocketRef.current.close();
            webSocketRef.current = null; // Clear ref after closing
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
        // Cleanup function for when the component unmounts or autoUpdate changes
        return () => {
            disconnectWebSocket();
        };
    }, [autoUpdate, connectWebSocket, disconnectWebSocket]);

    const fetchExperiments = useCallback(async () => {
        if (!user || user.role !== 'METEOROLOGIST') {
            setIsLoading(false);
            setExperiments([]); // Clear experiments if user is not authorized
            return;
        }
        setIsLoading(true);
        setError(null);
        try {
            const pageable = {
                page: pagination.page,
                size: pagination.size,
                sortBy: 'startTime', // Default sort
                sortDir: 'DESC'
            };
            const activeFilters = { ...filters };
            // Ensure dates are ISO strings if not null
            if (activeFilters.startedAfter) activeFilters.startedAfter = new Date(activeFilters.startedAfter).toISOString();
            if (activeFilters.finishedBefore) activeFilters.finishedBefore = new Date(activeFilters.finishedBefore).toISOString();

            const data = await experimentService.getExperiments(activeFilters, pageable);
            setExperiments(data.content || []);
            setPagination(prev => ({
                ...prev,
                totalPages: data.totalPages,
                totalElements: data.totalElements,
            }));
        } catch (err) {
            setError(err.response?.data?.message || err.message || 'Failed to fetch experiments.');
            setExperiments([]);
            setPagination(prev => ({ ...prev, totalPages: 0, totalElements: 0 }));
        } finally {
            setIsLoading(false);
        }
    }, [user, filters, pagination.page, pagination.size]); // Dependencies for useCallback

    useEffect(() => {
        fetchExperiments();
    }, [fetchExperiments]); // fetchExperiments is memoized by useCallback

    const handleFilterChange = (event) => {
        const { name, value } = event.target;
        setFilters(prev => ({ ...prev, [name]: value }));
        setPagination(prev => ({ ...prev, page: 0 })); // Reset to first page on filter change
    };

    const handleDateChange = (name, date) => {
        setFilters(prev => ({ ...prev, [name]: date })); // Store Date object, convert to ISO on fetch
        setPagination(prev => ({ ...prev, page: 0 }));
    };

    const handleClearFilters = () => {
        setFilters({ modelType: '', datasetName: '', status: '', startedAfter: null, finishedBefore: null });
        setPagination(prev => ({ ...prev, page: 0 }));
        // fetchExperiments will be re-triggered by useEffect due to filter state change
    };

    const handlePageChange = (event, value) => {
        setPagination(prev => ({ ...prev, page: value - 1 })); // MUI Pagination is 1-indexed
    };

    const openDeleteDialog = (experimentId, experimentName) => {
        setDeleteConfirm({ open: true, experimentId, experimentName });
    };

    const handleConfirmDelete = async () => {
        if (deleteConfirm.experimentId) {
            setError(null); // Clear previous errors
            try {
                await experimentService.deleteExperiment(deleteConfirm.experimentId);
                setDeleteConfirm({ open: false, experimentId: null, experimentName: '' });
                fetchExperiments(); // Refresh list
            } catch (err) {
                setError(err.response?.data?.message || err.message || `Failed to delete experiment ${deleteConfirm.experimentName}.`);
                setDeleteConfirm({ open: false, experimentId: null, experimentName: '' });
            }
        }
    };

    if (!user) return <LoadingSpinner />; // Or redirect to login if AuthProvider handles it
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
                    <Typography variant="h4" component="h1">
                        Experiments Dashboard
                    </Typography>
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
                          <span> {/* Span for Tooltip when button is disabled */}
                              <IconButton onClick={() => fetchExperiments(true)} disabled={isLoading} color="primary">
                              <RefreshIcon />
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Button
                            variant="outlined"
                            startIcon={<FilterListIcon />}
                            onClick={() => setShowFilters(!showFilters)}
                            sx={{ ml: 2, mr: 2 }}
                        >
                            {showFilters ? 'Hide Filters' : 'Show Filters'}
                        </Button>
                        <Button
                            variant="contained"
                            startIcon={<AddIcon />}
                            onClick={() => navigate('/experiments/create')}
                        >
                            New Experiment
                        </Button>
                    </Box>
                </Box>

                {showFilters && (
                    <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
                        <Grid container spacing={2} alignItems="flex-end"> {/* alignItems to bottom align button */}
                            <Grid item xs={12} sm={6} md={3} lg={2}>
                                <TextField fullWidth label="Model Type" name="modelType" value={filters.modelType} onChange={handleFilterChange} variant="outlined" size="small"/>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3} lg={2}>
                                <TextField fullWidth label="Dataset Name" name="datasetName" value={filters.datasetName} onChange={handleFilterChange} variant="outlined" size="small"/>
                            </Grid>
                            <Grid item xs={12} sm={6} md={2} lg={2}>
                                <FormControl fullWidth variant="outlined" size="small">
                                    <InputLabel>Status</InputLabel>
                                    <Select name="status" value={filters.status} label="Status" onChange={handleFilterChange}>
                                        <MenuItem value=""><em>Any</em></MenuItem>
                                        <MenuItem value="PENDING">Pending</MenuItem>
                                        <MenuItem value="RUNNING">Running</MenuItem>
                                        <MenuItem value="COMPLETED">Completed</MenuItem>
                                        <MenuItem value="FAILED">Failed</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3} lg={2.5}>
                                <DateTimePicker
                                    label="Started After"
                                    value={filters.startedAfter}
                                    onChange={(newValue) => handleDateChange('startedAfter', newValue)}
                                    slotProps={{ textField: { size: 'small', fullWidth: true, variant: 'outlined' } }}
                                />
                            </Grid>
                            <Grid item xs={12} sm={6} md={3} lg={2.5}>
                                <DateTimePicker
                                    label="Finished Before"
                                    value={filters.finishedBefore}
                                    onChange={(newValue) => handleDateChange('finishedBefore', newValue)}
                                    slotProps={{ textField: { size: 'small', fullWidth: true, variant: 'outlined' } }}
                                />
                            </Grid>
                            <Grid item xs={12} sm={6} md={1} lg={1} sx={{display:'flex', alignItems:'flex-end'}}>
                                <Button onClick={handleClearFilters} variant="text" size="medium">Clear</Button>
                            </Grid>
                        </Grid>
                    </Paper>
                )}

                {isLoading && <Box sx={{display: 'flex', justifyContent: 'center', my: 5}}><CircularProgress /></Box>}
                {!isLoading && error && <Alert severity="error" sx={{my: 2}}>{error}</Alert>}
                {!isLoading && !error && (
                    <>
                        {experiments.length > 0 ? (
                            <ExperimentGrid experiments={experiments} onDelete={openDeleteDialog} />
                        ) : (
                            <Paper sx={{p:3, textAlign:'center', mt:3}}><Typography>No experiments found matching your criteria.</Typography></Paper>
                        )}
                        {experiments.length > 0 && pagination.totalPages > 1 && (
                            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb:2 }}>
                                <Pagination
                                    count={pagination.totalPages}
                                    page={pagination.page + 1} // MUI Pagination is 1-indexed
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
                onClose={() => setDeleteConfirm({ open: false, experimentId: null, experimentName: '' })}
                onConfirm={handleConfirmDelete}
                title="Delete Experiment?"
                message={`Are you sure you want to delete the experiment "${deleteConfirm.experimentName}" (ID: ${deleteConfirm.experimentId})? This action cannot be undone and will also attempt to delete associated artifacts.`}
                confirmText="Delete"
            />
        </LocalizationProvider>
    );
};

export default ExperimentsPage;