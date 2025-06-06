package ro.ubb.ai.javaserver.repository;

import ro.ubb.ai.javaserver.entity.Image;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface ImageRepository extends JpaRepository<Image, Long> {
    List<Image> findByUserIdOrderByIdDesc(Long userId); // Or OrderByUploadedAtDesc
}
