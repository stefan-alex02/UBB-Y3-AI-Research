import React, {useEffect, useState} from 'react';
import {Box, Button, Container, Typography} from '@mui/material';
import ImageGrid from '../components/ImageGrid/ImageGrid';
import imageService from '../services/imageService';
import useAuth from '../hooks/useAuth';
import LoadingSpinner from '../components/LoadingSpinner';
import FileUploadButton from '../components/FileUploadButton';
import NewPredictionModal from "../components/Modals/NewPredictionModal";
import AssessmentIcon from "@mui/icons-material/Assessment";

const UploadedImagesPage = () => {
    const [images, setImages] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const { user } = useAuth();

    const [selectedImageIds, setSelectedImageIds] = useState(new Set());
    const [newPredictionModalOpen, setNewPredictionModalOpen] = useState(false);

    const fetchImages = async () => {
        setIsLoading(true);
        setError(null);
        try {
            if (user) { // Ensure user is loaded
                const data = await imageService.getUserImages();
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
    }, [user]);

    const handleImageUploadSuccess = () => {
        fetchImages();
    };

    const handleImageDelete = async (imageId) => {
        try {
            await imageService.deleteImage(imageId);
            setImages(prevImages => prevImages.filter(img => img.id !== imageId));
        } catch (err) {
            setError(err.message || 'Failed to delete image.');
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
        setSelectedImageIds(new Set());
    };

    if (isLoading) return <LoadingSpinner />;
    if (error) return <Typography color="error">Error: {error}</Typography>;

    return (
        <Container maxWidth="lg">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" component="h1">Your Uploaded Images</Typography>
                <Box sx={{ '& > *:not(:last-child)': { mr: 1 } }}>
                    <FileUploadButton onUploadSuccess={handleImageUploadSuccess} />
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
                selectedImageIds={selectedImageIds}
                onImageSelect={handleImageSelectToggle}
            />
            {newPredictionModalOpen && selectedImageIds.size > 0 && (
                <NewPredictionModal
                    open={newPredictionModalOpen}
                    onClose={() => setNewPredictionModalOpen(false)}
                    imageIds={Array.from(selectedImageIds)}
                    onPredictionCreated={handlePredictionCreated}
                />
            )}
        </Container>
    );
};

export default UploadedImagesPage;