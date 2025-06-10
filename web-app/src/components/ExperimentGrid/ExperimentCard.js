import React, { useState } from 'react';
import { Card, CardContent, CardActions, Typography, Button, Chip, Box, IconButton, Menu, MenuItem } from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science'; // Assuming you might use this if modelType is generic
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import DatasetIcon from '@mui/icons-material/Dataset';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import { useNavigate } from 'react-router-dom';
import ConfirmDialog from '../ConfirmDialog';
import {formatDateSafe} from "../../utils/dateUtils";

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
        onDelete(experiment.experiment_run_id); // Use snake_case
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
                            {experiment.name || 'Unnamed Experiment'} {/* Use name directly */}
                        </Typography>
                        <IconButton size="small" onClick={handleMenuClick}>
                            <MoreVertIcon />
                        </IconButton>
                        <Menu anchorEl={anchorEl} open={openMenu} onClose={handleMenuClose}>
                            <MenuItem onClick={() => { navigate(`/experiments/${experiment.experiment_run_id}`); handleMenuClose(); }}>
                                <VisibilityIcon sx={{ mr: 1 }} /> View Details
                            </MenuItem>
                            <MenuItem onClick={handleDeleteClick} sx={{ color: 'error.main' }}>
                                <DeleteIcon sx={{ mr: 1 }} /> Delete
                            </MenuItem>
                        </Menu>
                    </Box>
                    <Chip
                        icon={<ModelTrainingIcon />}
                        label={`Model: ${experiment.model_type || 'N/A'}`}
                        size="small"
                        sx={{ mb: 1, mr: 1 }}
                        clickable={false}
                        onClick={() => {}}
                    />
                    <Chip
                        icon={<DatasetIcon />}
                        label={`Dataset: ${experiment.dataset_name || 'N/A'}`}
                        size="small"
                        sx={{ mb: 1 }}
                        clickable={false}
                        onClick={() => {}}
                    />
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1, wordBreak: 'break-all' }}>
                        Run ID: <Typography component="span" variant="caption" sx={{ fontFamily: 'monospace' }}>{experiment.experiment_run_id || 'N/A'}</Typography>
                    </Typography>
                    <Typography component="div" variant="body2" color="text.secondary"> {/* Ensure parent is not unintentionally clickable */}
                        Status:{" "}
                        <Chip
                            label={experiment.status || 'N/A'}
                            color={getStatusColor(experiment.status)}
                            size="small"
                            clickable={false}
                            onClick={() => {}}
                        />
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Started: {formatDateSafe(experiment.start_time)} {/* Use snake_case & formatDate */}
                    </Typography>
                    {experiment.end_time && (
                        <Typography variant="body2" color="text.secondary">
                            Ended: {formatDateSafe(experiment.end_time)} {/* Use snake_case & formatDate */}
                        </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary">
                        By: {experiment.user_name || 'N/A'} {/* Use snake_case */}
                    </Typography>
                </CardContent>
                <CardActions>
                    <Button
                        size="small"
                        startIcon={<VisibilityIcon />}
                        onClick={() => navigate(`/experiments/${experiment.experiment_run_id}`)} // Use snake_case
                    >
                        View Results
                    </Button>
                </CardActions>
            </Card>
            <ConfirmDialog
                open={dialogOpen} // Controlled by dialogOpen state
                onClose={() => setDialogOpen(false)}
                onConfirm={handleConfirmDelete}
                title="Delete Experiment?"
                message={`Are you sure you want to delete the experiment "${experiment.name || 'N/A'}" (ID: ${experiment.experiment_run_id || 'N/A'})? This action cannot be undone and will also attempt to delete associated artifacts.`}
                // CORRECTED: Use `experiment.name` and `experiment.experiment_run_id` directly from the prop
                // INSTEAD OF: deleteConfirm.experimentName and deleteConfirm.experimentId
                confirmText="Delete"
            />
        </>
    );
};

export default ExperimentCard;