package ro.ubb.ai.javaserver.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import ro.ubb.ai.javaserver.dto.image.ImageDTO;
import ro.ubb.ai.javaserver.exception.ResourceNotFoundException;
import ro.ubb.ai.javaserver.service.ImageService;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/images")
@RequiredArgsConstructor
public class ImageController {

    private final ImageService imageService;

    @PostMapping("/upload")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<ImageDTO> uploadImage(@RequestParam("file") MultipartFile file) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        ImageDTO uploadedImage = imageService.uploadImage(file, currentUsername);
        return new ResponseEntity<>(uploadedImage, HttpStatus.CREATED);
    }

    @GetMapping
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<List<ImageDTO>> getUserImages() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        return ResponseEntity.ok(imageService.getImagesForUser(currentUsername));
    }

    @GetMapping("/{imageId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<ImageDTO> getImage(@PathVariable Long imageId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        return ResponseEntity.ok(imageService.getImageByIdForUser(imageId, currentUsername));
    }

    @GetMapping("/{imageId}/content")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<byte[]> getImageContent(@PathVariable Long imageId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();

        try {
            byte[] imageBytes = imageService.getImageContent(imageId, currentUsername);

            ImageDTO imageDetails = imageService.getImageByIdForUser(imageId, currentUsername);

            HttpHeaders headers = new HttpHeaders();
            String contentType = "application/octet-stream";
            if (imageDetails != null && imageDetails.getFormat() != null) {
                String format = imageDetails.getFormat().toLowerCase();
                if ("png".equals(format)) contentType = MediaType.IMAGE_PNG_VALUE;
                else if ("jpg".equals(format) || "jpeg".equals(format)) contentType = MediaType.IMAGE_JPEG_VALUE;
                else if ("gif".equals(format)) contentType = MediaType.IMAGE_GIF_VALUE;
            }
            headers.setContentType(MediaType.parseMediaType(contentType));
            headers.setContentLength(imageBytes.length);

            return new ResponseEntity<>(imageBytes, headers, HttpStatus.OK);

        } catch (ResourceNotFoundException e) {
            log.warn("Image content not found for imageId {}: {}", imageId, e.getMessage());
            return ResponseEntity.notFound().build();
        } catch (SecurityException e) {
            log.warn("Security exception for image content imageId {}: {}", imageId, e.getMessage());
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        } catch (Exception e) {
            log.error("Error serving image content for imageId {}: {}", imageId, e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @DeleteMapping("/{imageId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Void> deleteImage(@PathVariable Long imageId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        imageService.deleteImage(imageId, currentUsername);
        return ResponseEntity.noContent().build();
    }
}
