import React, {useState} from 'react';
import {Box, Button, Card, CardActions, CardContent, Chip, IconButton, Menu, MenuItem, Typography} from '@mui/material';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import DatasetIcon from '@mui/icons-material/Dataset';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import {useNavigate} from 'react-router-dom';
import {formatDateSafe} from "../../utils/dateUtils";

const ExperimentCard = ({ experiment, onDeleteRequest }) => {
    const navigate = useNavigate();
    const [anchorEl, setAnchorEl] = useState(null);
    const openMenu = Boolean(anchorEl);

    const handleMenuClick = (event) => {
        event.stopPropagation();
        setAnchorEl(event.currentTarget);
    };

    const handleMenuClose = (event) => {
        if (event) event.stopPropagation();
        setAnchorEl(null);
    };

    const handleDeleteInitiate = (event) => {
        event.stopPropagation();
        if (onDeleteRequest) {
            onDeleteRequest(experiment.experiment_run_id, experiment.name);
        }
        handleMenuClose(event); // Close the menu
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
                            {experiment.name || 'Unnamed Experiment'}
                        </Typography>
                        <IconButton size="small" onClick={handleMenuClick}>
                            <MoreVertIcon />
                        </IconButton>
                        <Menu anchorEl={anchorEl} open={openMenu} onClose={handleMenuClose}>
                            <MenuItem onClick={() => { navigate(`/experiments/${experiment.experiment_run_id}`); handleMenuClose(); }}>
                                <VisibilityIcon sx={{ mr: 1 }} /> View Details
                            </MenuItem>
                            <MenuItem onClick={handleDeleteInitiate} sx={{ color: 'error.main' }}>
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
                    <Typography component="div" variant="body2" color="text.secondary">
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
                        Started: {formatDateSafe(experiment.start_time)}
                    </Typography>
                    {experiment.end_time && (
                        <Typography variant="body2" color="text.secondary">
                            Ended: {formatDateSafe(experiment.end_time)}
                        </Typography>
                    )}
                    <Typography variant="body2" color="text.secondary">
                        By: {experiment.user_name || 'N/A'}
                    </Typography>
                </CardContent>
                <CardActions>
                    <Button
                        size="small"
                        startIcon={<VisibilityIcon />}
                        onClick={() => navigate(`/experiments/${experiment.experiment_run_id}`)}
                    >
                        View Results
                    </Button>
                </CardActions>
            </Card>
        </>
    );
};

export default ExperimentCard;