import React from 'react';
import {Paper, Typography, Box, Alert} from '@mui/material';

const JsonViewer = ({ jsonData, title }) => {
    let displayData = jsonData;
    let parseError = null;

    if (typeof jsonData === 'string') {
        try {
            displayData = JSON.parse(jsonData);
        } catch (e) {
            console.error("JsonViewer: Error parsing JSON string:", e, "Data:", jsonData);
            parseError = `Error parsing JSON: ${e.message}`;
            displayData = jsonData; // Show raw string on error
        }
    } else if (typeof jsonData !== 'object' || jsonData === null) {
        // If it's not a string and not an object, it's unexpected
        parseError = "Invalid data type for JSON viewer. Expected object or JSON string.";
        displayData = String(jsonData); // Show as string
    }
    // If it's already an object, displayData is already jsonData

    return (
        <Paper elevation={0} sx={{ p: 1, my: 0, height: '100%', display:'flex', flexDirection:'column', overflow:'hidden' }}>
            {title && <Typography variant="subtitle2" gutterBottom sx={{px:1, pt:1, flexShrink:0}}>{title}</Typography>}
            {parseError && <Alert severity="error" sx={{m:1, flexShrink:0}}>{parseError}</Alert>}
            <Box component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '0.8rem', flexGrow:1, overflow:'auto', m:0, p:1, backgroundColor: theme => theme.palette.mode === 'dark' ? theme.palette.grey[800] : theme.palette.grey[100], borderRadius: theme => theme.shape.borderRadius / 2 }}>
                {typeof displayData === 'object' ? JSON.stringify(displayData, null, 2) : displayData}
            </Box>
        </Paper>
    );
};
export default JsonViewer;