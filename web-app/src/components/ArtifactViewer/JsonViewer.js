import React from 'react';
import { Paper, Typography, Box } from '@mui/material';

const JsonViewer = ({ jsonData, title }) => {
    let displayData = jsonData;
    if (typeof jsonData === 'string') {
        try {
            displayData = JSON.parse(jsonData);
        } catch (e) {
            return <Typography color="error">Error parsing JSON data.</Typography>;
        }
    }

    return (
        <Paper elevation={1} sx={{ p: 2, my: 1, maxHeight: '400px', overflow: 'auto', backgroundColor: 'background.default' }}>
            {title && <Typography variant="subtitle1" gutterBottom>{title}</Typography>}
            <Box component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '0.8rem' }}>
                {JSON.stringify(displayData, null, 2)}
            </Box>
        </Paper>
    );
};
export default JsonViewer;