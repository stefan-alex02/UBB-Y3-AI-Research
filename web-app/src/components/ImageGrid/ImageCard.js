import React, { useState, useEffect } from 'react';
import {
    Card,
    CardMedia,
    CardContent,
    CardActions,
    Typography,
    Button,
    IconButton,
    Box,
    Skeleton,
    Checkbox, FormControlLabel
} from '@mui/material'; // Added Skeleton
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import BrokenImageIcon from '@mui/icons-material/BrokenImage'; // For error state
import { useNavigate } from 'react-router-dom';
import imageService from '../../services/imageService'; // Import your service
import ConfirmDialog from '../ConfirmDialog';
import {formatDateSafe} from "../../utils/dateUtils";


const ImageCard = ({ image, onDelete, selectedImageIds, onImageSelect }) => {
    const navigate = useNavigate();
    const [imageUrl, setImageUrl] = useState(null);
    const [isLoadingImage, setIsLoadingImage] = useState(true);
    const [imageError, setImageError] = useState(false);
    const [confirmOpen, setConfirmOpen] = useState(false);

    const isSelected = selectedImageIds.has(image.id);

    useEffect(() => {
        let objectUrl = null;
        const loadImage = async () => {
            if (image && image.id) {
                setIsLoadingImage(true);
                setImageError(false);
                try {
                    const blob = await imageService.getImageContentBlob(image.id);
                    objectUrl = URL.createObjectURL(blob);
                    setImageUrl(objectUrl);
                } catch (error) {
                    console.error("Failed to load image content for ID:", image.id, error);
                    setImageError(true);
                } finally {
                    setIsLoadingImage(false);
                }
            }
        };

        loadImage();

        return () => {
            // Cleanup: Revoke the object URL when the component unmounts or image changes
            if (objectUrl) {
                URL.revokeObjectURL(objectUrl);
            }
        };
    }, [image]); // Re-fetch if the image prop changes

    const handleViewPredictions = () => {
        navigate(`/images/${image.id}/predictions`);
    };

    const handleDeleteClick = () => {
        setConfirmOpen(true);
    };

    const handleConfirmDelete = () => {
        onDelete(image.id);
        setConfirmOpen(false);
    };


    return (
        <>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative', border: isSelected ? '2px solid' : '1px solid', borderColor: isSelected ? 'primary.main' : 'transparent' }}>
                {isLoadingImage ? (
                    <Skeleton variant="rectangular" width="100%" height={180} animation="wave" />
                ) : imageError ? (
                    <Box sx={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'grey.200' }}>
                        <BrokenImageIcon color="action" sx={{ fontSize: 60 }} />
                    </Box>
                ) : (
                    <>
                        <FormControlLabel
                            sx={{ position: 'absolute', zIndex: 1, backgroundColor: 'rgba(0,0,0,0.3)', borderRadius:1, p:0.5 }}
                            control={
                                <Checkbox
                                    size="small"
                                    checked={isSelected}
                                    onChange={() => onImageSelect(image.id)}
                                    onClick={(e) => e.stopPropagation()} // Prevent card click when toggling checkbox
                                />
                            }
                        />
                        <CardMedia
                            component="img"
                            sx={{ height: 180, objectFit: 'cover' }}
                            image={imageUrl}
                            alt={`Uploaded image ${image.id}`}
                        />
                    </>
                )}
                <CardContent sx={{ flexGrow: 1 }}>
                    <Typography gutterBottom variant="h6" component="div" noWrap>
                        Image ID: {image.id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Format: {image.format?.toUpperCase()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Uploaded: {formatDateSafe(image.uploaded_at)}
                    </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: 'space-between', p: 1 }}>
                    <Button size="small" startIcon={<VisibilityIcon />} onClick={handleViewPredictions}>
                        Predictions
                    </Button>
                    <IconButton size="small" color="error" onClick={handleDeleteClick}>
                        <DeleteIcon />
                    </IconButton>
                </CardActions>
            </Card>
            <ConfirmDialog
                open={confirmOpen}
                onClose={() => setConfirmOpen(false)}
                onConfirm={handleConfirmDelete}
                title="Delete Image?"
                message={`Are you sure you want to delete Image ID: ${image.id}? This will also remove any associated predictions.`}
                confirmText="Delete"
            />
        </>
    );
};

export default ImageCard;