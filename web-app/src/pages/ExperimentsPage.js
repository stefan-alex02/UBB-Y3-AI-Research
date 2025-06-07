// src/pages/ExperimentsPage.js
import React, { useState, useEffect, useCallback } from 'react';
import {
    Container, Typography, Button, Box, Grid, TextField, Select, MenuItem,
    FormControl, InputLabel, CircularProgress, Paper, Pagination, Alert, IconButton
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import FilterListIcon from '@mui/icons-material/FilterList'; // For filter section toggle
import { useNavigate } from 'react-router-dom';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns'; // Or AdapterDayjs
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';

import ExperimentGrid from '../components/ExperimentGrid/ExperimentGrid';
import experimentService from '../services/experimentService';
import useAuth from '../hooks/useAuth';
import LoadingSpinner from '../components/LoadingSpinner';
import ConfirmDialog from '../components/ConfirmDialog'; // Assuming this is in components/

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
                    <Box>
                        <Button
                            variant="outlined"
                            startIcon={<FilterListIcon />}
                            onClick={() => setShowFilters(!showFilters)}
                            sx={{ mr: 2 }}
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