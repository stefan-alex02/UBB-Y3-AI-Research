import React from 'react';
import {Box, CircularProgress, Dialog, DialogContent, DialogTitle, IconButton, Typography} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const ImageFullscreenModal = ({ open, onClose, imageUrl, imageBlob, title, altText = "Fullscreen Image" }) => {
    const [displayUrl, setDisplayUrl] = React.useState(null);
    const [isLoading, setIsLoading] = React.useState(false);

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
                setIsLoading(false);
            } else {
                setDisplayUrl(null);
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


    if (!open || (!displayUrl && !isLoading)) return null;

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="xl"
            fullWidth
            PaperProps={{
                sx: {
                    backgroundColor: 'rgba(0,0,0,0.75)',
                    height: '100vh',
                    maxHeight: '100vh',
                    m: 0,
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
            {!title && (
                <IconButton
                    aria-label="close"
                    onClick={onClose}
                    sx={{
                        position: 'absolute',
                        right: 8,
                        top: 8,
                        zIndex: 1,
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
                            maxHeight: 'calc(100vh - 64px - 16px)',
                            objectFit: 'contain',
                            display: 'block',
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