import React, {useState} from 'react';
import {Box, CircularProgress, IconButton, Paper, Tooltip, Typography} from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import BrokenImageIcon from '@mui/icons-material/BrokenImage';

const PlotViewer = ({ artifactUrl, altText, title, onZoom }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(false);

    React.useEffect(() => {
        setLoading(true);
        setError(false);
    }, [artifactUrl]);

    return (
        <Paper
            elevation={0}
            sx={{
                p: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'flex-start',
                position: 'relative',
                height: '100%',
                backgroundColor: 'transparent',
            }}
        >
            {title && (
                <Typography
                    variant="subtitle2"
                    gutterBottom
                    align="center"
                    sx={{
                        p: 1,
                        width: '100%',
                        flexShrink: 0,
                        borderBottom: (theme) => `1px solid ${theme.palette.divider}`,
                        mb: 1,
                    }}
                >
                    {title}
                </Typography>
            )}

            <Box sx={{
                position: 'relative',
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                maxHeight: '100%',
            }}>
                {loading && !error && <CircularProgress sx={{my:2}}/>}
                {!loading && error && (
                    <Box sx={{textAlign: 'center', color: 'text.secondary'}}>
                        <BrokenImageIcon sx={{fontSize: '3rem', mb:1}}/>
                        <Typography>Image failed to load.</Typography>
                        <Typography variant="caption">{altText}</Typography>
                    </Box>
                )}
                <Box
                    component="img"
                    src={artifactUrl}
                    alt={altText || 'Plot'}
                    sx={{
                        maxWidth: '100%',
                        maxHeight: '100%',
                        objectFit: 'contain',
                        display: loading || error ? 'none' : 'block',
                    }}
                    onLoad={() => setLoading(false)}
                    onError={() => { setLoading(false); setError(true); }}
                />
                {!loading && !error && artifactUrl && onZoom && (
                    <Tooltip title="View Fullscreen">
                        <IconButton
                            onClick={onZoom}
                            size="small"
                            sx={{
                                position: 'absolute',
                                top: 8,
                                right: 8,
                                backgroundColor: 'rgba(0,0,0,0.4)',
                                color: 'white',
                                '&:hover': {
                                    backgroundColor: 'rgba(0,0,0,0.6)',
                                },
                                zIndex: 1,
                            }}
                        >
                            <ZoomInIcon fontSize="inherit" />
                        </IconButton>
                    </Tooltip>
                )}
            </Box>
        </Paper>
    );
};
export default PlotViewer;