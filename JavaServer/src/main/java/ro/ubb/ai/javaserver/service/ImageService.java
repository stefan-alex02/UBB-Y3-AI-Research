package ro.ubb.ai.javaserver.service;

import org.springframework.web.multipart.MultipartFile;
import ro.ubb.ai.javaserver.dto.image.ImageDTO;

import java.util.List;

public interface ImageService {
    ImageDTO uploadImage(MultipartFile file, String username);

    List<ImageDTO> getImagesForUser(String username);
    ImageDTO getImageByIdForUser(Long imageId, String username);
    byte[] getImageContent(Long imageId, String username);

    void deleteImage(Long imageId, String username);
}
