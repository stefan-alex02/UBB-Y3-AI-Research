import React, {useState, useEffect} from 'react';
import { Paper, Box, CircularProgress, Typography } from '@mui/material';

const PlotViewer = ({ artifactUrl, altText, title }) => {
    // artifactUrl should be the direct URL to the image (e.g., from MinIO via Java/Python proxy)
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(false);

    return (
        <Paper elevation={1} sx={{ p: 1, my: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {title && <Typography variant="subtitle1" gutterBottom align="center">{title}</Typography>}
            {loading && <CircularProgress sx={{my:2}}/>}
            <img
                src={artifactUrl}
                alt={altText || 'Plot'}
                style={{ maxWidth: '100%', maxHeight: '500px', display: loading ? 'none' : 'block', objectFit: 'contain' }}
                onLoad={() => setLoading(false)}
                onError={() => { setLoading(false); setError(true); }}
            />
            {error && <Typography color="error" sx={{mt:1}}>Could not load image.</Typography>}
        </Paper>
    );
};
export default PlotViewer;