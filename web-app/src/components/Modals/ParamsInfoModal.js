import React from 'react';
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Button,
    Chip,
    Dialog,
    DialogContent,
    DialogTitle,
    IconButton,
    List,
    ListItem,
    ListItemText,
    Typography
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {PARAM_INFO} from '../../pages/experimentConfig';
import DialogActions from "@mui/material/DialogActions";

const ParamsInfoModal = ({ open, onClose, currentMethodName, currentModelType }) => {
    const getRelevantParams = () => {
        let relevant = [...PARAM_INFO.common_skorch];
        if (currentModelType === 'cnn' || currentModelType === 'stfeat' || currentModelType === 'cnn_feat') {
            relevant = relevant.concat(PARAM_INFO.module_cnn || []);
        } else if (currentModelType === 'pvit' || currentModelType === 'svit' || currentModelType === 'hyvit' || currentModelType === 'swin') {
            relevant = relevant.concat(PARAM_INFO.module_vit || []);
        }
        if (currentMethodName === 'non_nested_grid_search' || currentMethodName === 'nested_grid_search') {
            relevant = relevant.concat(PARAM_INFO.grid_search_specific || []);
        }
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

                <Accordion defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle1">Relevant Parameters for "{currentMethodName}" with "{currentModelType}"</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <List dense>
                            {relevantParams.map(param => (
                                <ListItem key={param.key} divider>
                                    <ListItemText
                                        primary={<><Chip label={param.key} size="small" sx={{mr:1}} clickable={false} onClick={() => {}} />
                                            <Typography component="span" variant="caption" color="text.secondary">({param.type})</Typography></>}
                                        secondary={<>Example: <code>{param.example}</code> <br/> {param.description}</>}
                                    />
                                </ListItem>
                            ))}
                            {relevantParams.length === 0 && <ListItem><ListItemText primary="No specific parameter hints available for this combination." /></ListItem>}
                        </List>
                    </AccordionDetails>
                </Accordion>

            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
};

export default ParamsInfoModal;