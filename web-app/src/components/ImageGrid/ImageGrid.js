import React from 'react';
import { Grid, Typography, Box } from '@mui/material';
import ImageCard from './ImageCard'; // Create this component

const ImageGrid = ({ images, onDelete, selectedImageIds, onImageSelect }) => {
    if (!images || images.length === 0) {
        return (
            <Box sx={{ textAlign: 'center', mt: 5 }}>
                <Typography variant="subtitle1">No images uploaded yet.</Typography>
            </Box>
        );
    }

    return (
        <Grid container spacing={3}>
            {images.map((image) => (
                <Grid item key={image.id} xs={12} sm={6} md={4} lg={3}>
                    <ImageCard image={image}
                               onDelete={onDelete}
                               onImageSelect={onImageSelect}
                               selectedImageIds={selectedImageIds} />
                </Grid>
            ))}
        </Grid>
    );
};

export default ImageGrid;