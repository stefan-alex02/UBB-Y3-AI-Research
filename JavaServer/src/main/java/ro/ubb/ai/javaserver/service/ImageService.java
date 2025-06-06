package ro.ubb.ai.javaserver.service;

import ro.ubb.ai.javaserver.dto.image.ImageDTO;
import org.springframework.web.multipart.MultipartFile;
import java.util.List;

public interface ImageService {
    ImageDTO uploadImage(MultipartFile file, String username); // username for MinIO path
    List<ImageDTO> getImagesForUser(String username);
    ImageDTO getImageByIdForUser(Long imageId, String username);
    void deleteImage(Long imageId, String username);
    // byte[] downloadImage(Long imageId, String username); // Might be handled by Python API directly for frontend
}
