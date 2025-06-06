package ro.ubb.ai.javaserver.dto.prediction;

import lombok.Data;

@Data
public class ImageIdFormatPairDTO { // Corresponds to Python's ImageIdFormatPair
    private String imageId; // Using String for imageId from DB for flexibility
    private String imageFormat;
}
