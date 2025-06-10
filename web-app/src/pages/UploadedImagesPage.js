import React, { useState, useEffect } from 'react';
import { Container, Typography, Button, Box } from '@mui/material';
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate';
import ImageGrid from '../components/ImageGrid/ImageGrid';
import imageService from '../services/imageService'; // You'll create this
import useAuth from '../hooks/useAuth';
import LoadingSpinner from '../components/LoadingSpinner';
import FileUploadButton from '../components/FileUploadButton';
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";
import NewPredictionModal from "../components/Modals/NewPredictionModal";
import AssessmentIcon from "@mui/icons-material/Assessment"; // A new component

const UploadedImagesPage = () => {
    const [images, setImages] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const { user } = useAuth(); // To pass username if needed by service

    const [selectedImageIds, setSelectedImageIds] = useState(new Set()); // Store IDs of selected images
    const [newPredictionModalOpen, setNewPredictionModalOpen] = useState(false);

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

    const handleImageSelectToggle = (imageId) => {
        setSelectedImageIds(prevSelected => {
            const newSelected = new Set(prevSelected);
            if (newSelected.has(imageId)) {
                newSelected.delete(imageId);
            } else {
                newSelected.add(imageId);
            }
            return newSelected;
        });
    };

    const handleOpenNewPredictionModal = () => {
        if (selectedImageIds.size > 0) {
            setNewPredictionModalOpen(true);
        }
    };

    const handlePredictionCreated = () => {
        setNewPredictionModalOpen(false);
        setSelectedImageIds(new Set()); // Clear selection after submitting
        // Optionally, navigate or show a success message for batch prediction
        // fetchImages(); // Might not be necessary as individual predictions are created
    };

    if (isLoading) return <LoadingSpinner />;
    if (error) return <Typography color="error">Error: {error}</Typography>;

    return (
        <Container maxWidth="lg">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" component="h1">Your Uploaded Images</Typography>
                <Box>
                    <FileUploadButton onUploadSuccess={handleImageUploadSuccess} sx={{mr:2}} />
                    <Button
                        variant="contained"
                        color="secondary"
                        onClick={handleOpenNewPredictionModal}
                        disabled={selectedImageIds.size === 0}
                        startIcon={<AssessmentIcon />}
                    >
                        Predict Selected ({selectedImageIds.size})
                    </Button>
                </Box>
            </Box>
            <ImageGrid
                images={images}
                onDelete={handleImageDelete}
                selectedImageIds={selectedImageIds} // Pass selected IDs
                onImageSelect={handleImageSelectToggle} // Pass toggle handler
            />
            {newPredictionModalOpen && selectedImageIds.size > 0 && (
                <NewPredictionModal
                    open={newPredictionModalOpen}
                    onClose={() => setNewPredictionModalOpen(false)}
                    // Pass an array of image objects or just IDs and formats
                    // Let's pass image IDs and let the modal/service fetch formats if needed,
                    // or pass image objects if readily available and not too large for props.
                    // For simplicity, passing IDs. Java service can fetch formats.
                    imageIds={Array.from(selectedImageIds)}
                    onPredictionCreated={handlePredictionCreated}
                />
            )}
        </Container>
    );
};

export default UploadedImagesPage;