import React from 'react';
import {Box, Grid, Typography} from '@mui/material';
import ExperimentCard from './ExperimentCard';

const ExperimentGrid = ({ experiments, onDelete }) => {
    if (!experiments || experiments.length === 0) {
        return (
            <Box sx={{ textAlign: 'center', mt: 5 }}>
                <Typography variant="subtitle1">No experiments to display.</Typography>
            </Box>
        );
    }

    return (
        <Grid container spacing={3}>
            {experiments.map((exp) => (
                <Grid item key={exp.experiment_run_id} xs={12} sm={6} md={4}>
                    <ExperimentCard experiment={exp} onDeleteRequest={onDelete} />
                </Grid>
            ))}
        </Grid>
    );
};

export default ExperimentGrid;