import React, { useState, useEffect } from 'react';
import { Container, Typography, Button, Box } from '@mui/material';
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate';
import ImageGrid from '../components/ImageGrid/ImageGrid';
import imageService from '../services/imageService'; // You'll create this
import useAuth from '../hooks/useAuth';
import LoadingSpinner from '../components/LoadingSpinner';
import FileUploadButton from '../components/FileUploadButton'; // A new component

const UploadedImagesPage = () => {
    const [images, setImages] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const { user } = useAuth(); // To pass username if needed by service

    const fetchImages = async () => {
        setIsLoading(true);
        setError(null);
        try {
            if (user) { // Ensure user is loaded
                const data = await imageService.getUserImages(); // Service should get username from auth context or have it passed
                setImages(data);
            }
        } catch (err) {
            setError(err.message || 'Failed to fetch images.');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchImages();
    }, [user]); // Re-fetch if user changes (e.g., after login)

    const handleImageUploadSuccess = () => {
        fetchImages(); // Refresh the list after successful upload
    };

    const handleImageDelete = async (imageId) => {
        try {
            await imageService.deleteImage(imageId);
            setImages(prevImages => prevImages.filter(img => img.id !== imageId));
        } catch (err) {
            setError(err.message || 'Failed to delete image.');
            // Optionally, re-fetch all images to ensure consistency
        }
    };

    if (isLoading) return <LoadingSpinner />;
    if (error) return <Typography color="error">Error: {error}</Typography>;

    return (
        <Container maxWidth="lg">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" component="h1">
                    Your Uploaded Images
                </Typography>
                <FileUploadButton onUploadSuccess={handleImageUploadSuccess} />
            </Box>
            <ImageGrid images={images} onDelete={handleImageDelete} />
        </Container>
    );
};

export default UploadedImagesPage;