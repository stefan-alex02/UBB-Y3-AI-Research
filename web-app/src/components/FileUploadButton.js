import React, { useState } from 'react';
import { Button, CircularProgress, Alert } from '@mui/material';
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate';
import imageService from '../services/imageService';

const FileUploadButton = ({ onUploadSuccess }) => {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');

    const handleFileChange = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setUploading(true);
        setError('');

        try {
            await imageService.uploadImage(file); // Assumes service gets username from context/auth
            if (onUploadSuccess) {
                onUploadSuccess();
            }
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Upload failed.');
        } finally {
            setUploading(false);
            event.target.value = null; // Reset file input
        }
    };

    return (
        <>
            <Button
                variant="contained"
                component="label" // Makes the button act like a label for the hidden input
                startIcon={uploading ? <CircularProgress size={20} color="inherit" /> : <AddPhotoAlternateIcon />}
                disabled={uploading}
            >
                Upload Image
                <input type="file" hidden accept="image/png, image/jpeg, image/jpg" onChange={handleFileChange} />
            </Button>
            {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
        </>
    );
};

export default FileUploadButton;