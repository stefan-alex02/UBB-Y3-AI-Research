import React, { useState } from 'react';
import { Card, CardContent, CardActions, Typography, Button, Chip, Box, IconButton, Menu, MenuItem } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import DatasetIcon from '@mui/icons-material/Dataset';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import { useNavigate } from 'react-router-dom';
import ConfirmDialog from '../ConfirmDialog'; // Assuming you have this

const ExperimentCard = ({ experiment, onDelete }) => {
    const navigate = useNavigate();
    const [anchorEl, setAnchorEl] = useState(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const openMenu = Boolean(anchorEl);

    const handleMenuClick = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const handleMenuClose = () => {
        setAnchorEl(null);
    };

    const handleDeleteClick = () => {
        setDialogOpen(true);
        handleMenuClose();
    };

    const handleConfirmDelete = () => {
        onDelete(experiment.experimentRunId);
        setDialogOpen(false);
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'COMPLETED': return 'success';
            case 'RUNNING': return 'info';
            case 'PENDING': return 'warning';
            case 'FAILED': return 'error';
            default: return 'default';
        }
    };

    return (
        <>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Typography variant="h6" component="div" gutterBottom noWrap title={experiment.name}>
                            {experiment.name}
                        </Typography>
                        <IconButton size="small" onClick={handleMenuClick}>
                            <MoreVertIcon />
                        </IconButton>
                        <Menu anchorEl={anchorEl} open={openMenu} onClose={handleMenuClose}>
                            <MenuItem onClick={() => { navigate(`/experiments/${experiment.experimentRunId}`); handleMenuClose(); }}>
                                <VisibilityIcon sx={{ mr: 1 }} /> View Details
                            </MenuItem>
                            <MenuItem onClick={handleDeleteClick} sx={{ color: 'error.main' }}>
                                <DeleteIcon sx={{ mr: 1 }} /> Delete
                            </MenuItem>
                        </Menu>
                    </Box>
                    <Chip
                        icon={<ModelTrainingIcon />}
                        label={`Model: ${experiment.modelType}`}
                        size="small"
                        sx={{ mb: 1, mr: 1 }}
                    />
                    <Chip
                        icon={<DatasetIcon />}
                        label={`Dataset: ${experiment.datasetName}`}
                        size="small"
                        sx={{ mb: 1 }}
                    />
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Run ID: <Typography component="span" variant="caption" sx={{ fontFamily: 'monospace' }}>{experiment.experimentRunId}</Typography>
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Status: <Chip label={experiment.status} color={getStatusColor(experiment.status)} size="small" />
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Started: {new Date(experiment.startTime).toLocaleString()}
                    </Typography>
                    {experiment.endTime && (
                        <Typography variant="body2" color="text.secondary">
                            Ended: {new Date(experiment.endTime).toLocaleString()}
                        </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary">
                        By: {experiment.userName || 'N/A'}
                    </Typography>
                </CardContent>
                <CardActions>
                    <Button
                        size="small"
                        startIcon={<VisibilityIcon />}
                        onClick={() => navigate(`/experiments/${experiment.experimentRunId}`)}
                    >
                        View Results
                    </Button>
                </CardActions>
            </Card>
            <ConfirmDialog
                open={dialogOpen}
                onClose={() => setDialogOpen(false)}
                onConfirm={handleConfirmDelete}
                title="Delete Experiment?"
                message={`Are you sure you want to delete the experiment "${experiment.name}" (ID: ${experiment.experimentRunId})? This action cannot be undone and will also delete any associated prediction artifacts (if implemented).`}
                confirmText="Delete"
            />
        </>
    );
};

export default ExperimentCard;