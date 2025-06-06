// src/components/Modals/ParamsInfoModal.js
import React from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    IconButton,
    Typography,
    Box,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    List,
    ListItem,
    ListItemText,
    Chip,
    Button
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { PARAM_INFO } from '../../pages/experimentConfig';
import DialogActions from "@mui/material/DialogActions"; // Adjust path

const ParamsInfoModal = ({ open, onClose, currentMethodName, currentModelType }) => {
    // Filter/select relevant PARAM_INFO sections based on currentMethodName and currentModelType
    const getRelevantParams = () => {
        let relevant = [...PARAM_INFO.common_skorch];
        if (currentModelType === 'cnn' || currentModelType === 'stfeat' || currentModelType === 'cnn_feat') {
            relevant = relevant.concat(PARAM_INFO.module_cnn || []);
        } else if (currentModelType === 'pvit' || currentModelType === 'svit' || currentModelType === 'hyvit' || currentModelType === 'swin') {
            relevant = relevant.concat(PARAM_INFO.module_vit || []); // Assuming module_vit covers these
        }
        // Add params specific to the method itself (like param_grid for grid_search)
        if (currentMethodName === 'non_nested_grid_search' || currentMethodName === 'nested_grid_search') {
            relevant = relevant.concat(PARAM_INFO.grid_search_specific || []);
        }
        // Add more conditions for other methods/models
        return relevant;
    };

    const relevantParams = getRelevantParams();

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth scroll="paper">
            <DialogTitle sx={{ m: 0, p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                Parameter Information Helper
                <IconButton aria-label="close" onClick={onClose} sx={{p:0.5}}> <CloseIcon /> </IconButton>
            </DialogTitle>
            <DialogContent dividers>
                <Typography variant="body2" sx={{mb:2}}>
                    This shows common parameters for the selected method and model type.
                    For grid search methods, `param_grid` values should be arrays (e.g., `"lr": [0.001, 0.0001]`).
                    For fixed parameter methods, values are single items (e.g., `"lr": 0.001`).
                    All keys and string values must be in double quotes in raw JSON.
                </Typography>

                {/* Could group by PARAM_INFO sections if desired */}
                <Accordion defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle1">Relevant Parameters for "{currentMethodName}" with "{currentModelType}"</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <List dense>
                            {relevantParams.map(param => (
                                <ListItem key={param.key} divider>
                                    <ListItemText
                                        primary={<><Chip label={param.key} size="small" sx={{mr:1}} /> <Typography component="span" variant="caption" color="text.secondary">({param.type})</Typography></>}
                                        secondary={<>Example: <code>{param.example}</code> <br/> {param.description}</>}
                                    />
                                </ListItem>
                            ))}
                            {relevantParams.length === 0 && <ListItem><ListItemText primary="No specific parameter hints available for this combination." /></ListItem>}
                        </List>
                    </AccordionDetails>
                </Accordion>

                {/* Add more accordions for other general sections if PARAM_INFO is structured that way */}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
};

export default ParamsInfoModal;