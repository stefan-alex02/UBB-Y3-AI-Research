package ro.ubb.ai.javaserver.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import ro.ubb.ai.javaserver.entity.Prediction;

import java.util.List;
import java.util.Optional;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    List<Prediction> findByImageIdOrderByPredictionTimestampDesc(Long imageId);
    Optional<Prediction> findByImageIdAndModelExperimentExperimentRunId(Long imageId, String modelExperimentRunId);
}
