import React from 'react';
import { Card, CardMedia, CardContent, CardActions, Typography, Button, IconButton, Box } from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import { useNavigate } from 'react-router-dom';
import { API_BASE_URL } from '../../config'; // Assuming this is your Java API base
import useAuth from '../../hooks/useAuth';

const ImageCard = ({ image, onDelete }) => {
    const navigate = useNavigate();
    const { user } = useAuth(); // To get username for image URL

    const handleViewPredictions = () => {
        navigate(`/images/${image.id}/predictions`);
    };

    // Construct image URL. This might point to a Java endpoint that proxies to Python/MinIO,
    // or directly to Python/MinIO if CORS and auth allow.
    // For now, let's assume a Java proxy or a direct Python image serving endpoint.
    // Java needs to serve the image at API_BASE_URL/images/{username}/{imageId}.{format}
    // or Python needs to serve it (more likely for direct display).
    // Let's use the python-proxy endpoint defined earlier as an example.
    const imageUrl = user ? `${API_BASE_URL}/python-proxy-images/${user.username}/${image.id}.${image.format}` : '#';


    return (
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardMedia
                component="img"
                sx={{
                    // Adjust height or use aspectRatio
                    height: 180,
                    objectFit: 'cover', // Or 'contain'
                }}
                image={imageUrl}
                alt={`Uploaded image ${image.id}`}
                onError={(e) => { e.target.onerror = null; e.target.src="https://via.placeholder.com/300x180?text=Image+Not+Found"; }} // Fallback
            />
            <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h6" component="div" noWrap>
                    Image ID: {image.id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Format: {image.format?.toUpperCase()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Uploaded: {new Date(image.uploadedAt).toLocaleDateString()}
                </Typography>
            </CardContent>
            <CardActions sx={{ justifyContent: 'space-between', p:1}}>
                <Button size="small" startIcon={<VisibilityIcon />} onClick={handleViewPredictions}>
                    Predictions
                </Button>
                <IconButton size="small" color="error" onClick={() => onDelete(image.id)}>
                    <DeleteIcon />
                </IconButton>
            </CardActions>
        </Card>
    );
};

export default ImageCard;