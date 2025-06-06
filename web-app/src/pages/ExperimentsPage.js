import React, { useState, useEffect, useCallback } from 'react';
import {
    Container,
    Typography,
    Button,
    Box,
    Grid,
    TextField,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    CircularProgress,
    Paper,
    Pagination,
    Alert
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import { useNavigate } from 'react-router-dom';
import ExperimentGrid from '../components/ExperimentGrid/ExperimentGrid';
import experimentService from '../services/experimentService';
import useAuth from '../hooks/useAuth'; // For role checks

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
        // startedAfter: null, // For DateTimePickers (more complex MUI component)
        // finishedBefore: null,
    });
    const [pagination, setPagination] = useState({
        page: 0, // 0-indexed for API
        size: 9, // Number of items per page
        totalPages: 0,
    });

    const fetchExperiments = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            // Prepare pageable for Spring Data Pageable (page, size, sort)
            const pageable = { page: pagination.page, size: pagination.size, sortBy: 'startTime', sortDir: 'DESC' };
            const data = await experimentService.getExperiments(filters, pageable);
            setExperiments(data.content || []); // Spring Page wraps content in 'content'
            setPagination(prev => ({ ...prev, totalPages: data.totalPages }));
        } catch (err) {
            setError(err.message || 'Failed to fetch experiments.');
            setExperiments([]);
        } finally {
            setIsLoading(false);
        }
    }, [filters, pagination.page, pagination.size]);

    useEffect(() => {
        if (user && user.role === 'METEOROLOGIST') {
            fetchExperiments();
        }
    }, [fetchExperiments, user]);

    const handleFilterChange = (event) => {
        const { name, value } = event.target;
        setFilters(prev => ({ ...prev, [name]: value }));
        setPagination(prev => ({ ...prev, page: 0 })); // Reset to first page on filter change
    };

    const handlePageChange = (event, value) => {
        setPagination(prev => ({ ...prev, page: value - 1 })); // MUI Pagination is 1-indexed
    };

    const handleExperimentDelete = async (experimentRunId) => {
        // Implement confirm dialog before deleting
        try {
            // await experimentService.deleteExperiment(experimentRunId);
            console.warn("Experiment deletion service call not implemented yet");
            fetchExperiments(); // Refresh list
        } catch (err) {
            setError(err.message || "Failed to delete experiment");
        }
    };


    if (user?.role !== 'METEOROLOGIST') {
        return <Typography>Access Denied. This page is for meteorologists only.</Typography>;
    }

    return (
        <Container maxWidth="xl">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, mt: 2 }}>
                <Typography variant="h4" component="h1">
                    Experiments
                </Typography>
                <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => navigate('/experiments/create')}
                >
                    New Experiment
                </Button>
            </Box>

            <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>Filters</Typography>
                <Grid container spacing={2}>
                    <Grid item xs={12} sm={4}>
                        <TextField fullWidth label="Model Type (e.g., pvit)" name="modelType" value={filters.modelType} onChange={handleFilterChange} variant="outlined" size="small"/>
                    </Grid>
                    <Grid item xs={12} sm={4}>
                        <TextField fullWidth label="Dataset Name (e.g., CCSN)" name="datasetName" value={filters.datasetName} onChange={handleFilterChange} variant="outlined" size="small"/>
                    </Grid>
                    <Grid item xs={12} sm={4}>
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
                    {/* Add DateTimePickers for date filters if needed */}
                </Grid>
            </Paper>

            {isLoading && <Box sx={{display: 'flex', justifyContent: 'center', my: 5}}><CircularProgress /></Box>}
            {error && <Alert severity="error">{error}</Alert>}
            {!isLoading && !error && (
                <>
                    <ExperimentGrid experiments={experiments} onDelete={handleExperimentDelete} />
                    {experiments.length > 0 && pagination.totalPages > 1 && (
                        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                            <Pagination
                                count={pagination.totalPages}
                                page={pagination.page + 1} // MUI is 1-indexed
                                onChange={handlePageChange}
                                color="primary"
                            />
                        </Box>
                    )}
                    {!isLoading && experiments.length === 0 && <Typography sx={{textAlign: 'center', mt: 3}}>No experiments found matching your criteria.</Typography>}
                </>
            )}
        </Container>
    );
};

export default ExperimentsPage;