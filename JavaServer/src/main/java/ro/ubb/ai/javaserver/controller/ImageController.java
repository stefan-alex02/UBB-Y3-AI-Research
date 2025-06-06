package ro.ubb.ai.javaserver.controller;

import ro.ubb.ai.javaserver.dto.image.ImageDTO;
import ro.ubb.ai.javaserver.service.ImageService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

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
        // Service layer will check if the current user owns the image
        return ResponseEntity.ok(imageService.getImageByIdForUser(imageId, currentUsername));
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
