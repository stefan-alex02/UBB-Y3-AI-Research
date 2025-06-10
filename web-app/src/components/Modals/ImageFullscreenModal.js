import React from 'react';
import { Dialog, DialogContent, DialogTitle, IconButton, Box, Typography, CircularProgress } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const ImageFullscreenModal = ({ open, onClose, imageUrl, imageBlob, title, altText = "Fullscreen Image" }) => {
    // Can accept either a direct imageUrl string or an imageBlob
    const [displayUrl, setDisplayUrl] = React.useState(null);
    const [isLoading, setIsLoading] = React.useState(false); // For blob conversion

    React.useEffect(() => {
        let objectUrl = null;
        if (open) {
            if (imageUrl) {
                setDisplayUrl(imageUrl);
                setIsLoading(false);
            } else if (imageBlob instanceof Blob) {
                setIsLoading(true);
                objectUrl = URL.createObjectURL(imageBlob);
                setDisplayUrl(objectUrl);
                // Image loading itself is handled by the <img> tag's onLoad
                // but we set isLoading to false once the object URL is created
                // A slight delay might be needed if the blob is very large before it's ready
                // For simplicity, we assume createObjectURL is fast enough.
                setIsLoading(false);
            } else {
                setDisplayUrl(null); // No valid source
                setIsLoading(false);
            }
        }

        return () => {
            if (objectUrl) {
                URL.revokeObjectURL(objectUrl);
                setDisplayUrl(null);
            }
        };
    }, [open, imageUrl, imageBlob]);


    if (!open || (!displayUrl && !isLoading)) return null; // Don't render if not open or no valid source

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="xl"
            fullWidth
            PaperProps={{
                sx: {
                    backgroundColor: 'rgba(0,0,0,0.75)', // Darker overlay for better focus on image
                    height: '100vh', // Attempt to make dialog itself take full height
                    maxHeight: '100vh',
                    m: 0, // Remove default margins
                    display: 'flex',
                    flexDirection: 'column'
                }
            }}
        >
            {title && (
                <DialogTitle sx={{ color: 'white', pb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {title}
                    <IconButton aria-label="close" onClick={onClose} sx={{ color: 'white' }}>
                        <CloseIcon />
                    </IconButton>
                </DialogTitle>
            )}
            {!title && ( // Still provide a close button if no title
                <IconButton
                    aria-label="close"
                    onClick={onClose}
                    sx={{
                        position: 'absolute',
                        right: 8,
                        top: 8,
                        zIndex: 1, // Ensure it's above the image
                        color: (theme) => theme.palette.grey[300],
                        backgroundColor: 'rgba(0,0,0,0.5)',
                        '&:hover': {
                            backgroundColor: 'rgba(0,0,0,0.7)',
                        }
                    }}
                >
                    <CloseIcon />
                </IconButton>
            )}
            <DialogContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', p: 1, overflow: 'hidden', flexGrow: 1 }}>
                {isLoading ? (
                    <CircularProgress color="inherit" sx={{color: 'white'}} />
                ) : displayUrl ? (
                    <Box
                        component="img"
                        src={displayUrl}
                        alt={altText}
                        sx={{
                            maxWidth: '100%',
                            maxHeight: 'calc(100vh - 64px - 16px)', // Adjust for DialogTitle and padding
                            objectFit: 'contain',
                            display: 'block', // Removes extra space below img
                        }}
                    />
                ) : (
                    <Typography sx={{color: 'white'}}>No image source provided or image failed to load.</Typography>
                )}
            </DialogContent>
        </Dialog>
    );
};

export default ImageFullscreenModal;