import React, { useState } from 'react';
import { Paper, Box, CircularProgress, Typography, IconButton, Tooltip } from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import BrokenImageIcon from '@mui/icons-material/BrokenImage';

const PlotViewer = ({ artifactUrl, altText, title, onZoom }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(false);

    // Reset loading/error state if artifactUrl changes
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
                flexDirection: 'column', // Stack title and image vertically
                alignItems: 'center',    // Center items horizontally
                justifyContent: 'flex-start', // Align items to the start (top)
                position: 'relative',
                height: '100%',
                backgroundColor: 'transparent',
            }}
        >
            {title && (
                <Typography
                    variant="subtitle2"
                    gutterBottom // Adds some margin below
                    align="center"
                    sx={{
                        p: 1, // Padding around the title
                        width: '100%',
                        flexShrink: 0, // Prevent title from shrinking
                        borderBottom: (theme) => `1px solid ${theme.palette.divider}`, // Optional separator
                        mb: 1, // Margin below title/separator
                    }}
                >
                    {title}
                </Typography>
            )}

            <Box sx={{
                position: 'relative', // For zoom button
                width: '100%',
                // flexGrow: 1, // Allow this box to take remaining space
                // minHeight: 0, // Important for flex item to shrink if needed
                // The image will now dictate the height of this box up to its container's limit
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                // Add padding if you want space around the image within this box
                // p:1,
                overflow: 'hidden', // Hide parts of image if it's too big for its allocated space before zoom
                maxHeight: '100%', // Constrain to parent height
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
                        maxHeight: '100%', // Let image scale down to fit this box
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
                                zIndex: 1, // Ensure it's above the image
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