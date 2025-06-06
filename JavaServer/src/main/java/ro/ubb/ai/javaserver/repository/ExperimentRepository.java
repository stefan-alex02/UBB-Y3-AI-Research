package ro.ubb.ai.javaserver.repository;

import ro.ubb.ai.javaserver.entity.Experiment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor; // For filtering
import org.springframework.stereotype.Repository;

@Repository
public interface ExperimentRepository extends JpaRepository<Experiment, String>, JpaSpecificationExecutor<Experiment> {
    // JpaSpecificationExecutor allows dynamic query building for filters
}
