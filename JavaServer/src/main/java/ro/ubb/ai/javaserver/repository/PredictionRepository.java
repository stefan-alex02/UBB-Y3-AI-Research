package ro.ubb.ai.javaserver.repository;

import ro.ubb.ai.javaserver.entity.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    List<Prediction> findByImageIdOrderByPredictionTimestampDesc(Long imageId);
    Optional<Prediction> findByImageIdAndModelExperimentExperimentRunId(Long imageId, String modelExperimentRunId);
    void deleteByImageIdAndModelExperimentExperimentRunId(Long imageId, String modelExperimentRunId);
}
