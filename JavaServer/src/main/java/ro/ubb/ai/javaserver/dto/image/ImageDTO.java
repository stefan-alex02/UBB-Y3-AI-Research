package ro.ubb.ai.javaserver.dto.image;

import lombok.Data;

import java.time.OffsetDateTime;

@Data
public class ImageDTO {
    private Long id;
    private String format;
    private Long userId;
    private OffsetDateTime uploadedAt;
    private String presignedUrl;
}