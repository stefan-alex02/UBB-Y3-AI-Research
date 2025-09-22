package ro.ubb.ai.javaserver.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;
import ro.ubb.ai.javaserver.entity.Experiment;

@Repository
public interface ExperimentRepository extends JpaRepository<Experiment, String>, JpaSpecificationExecutor<Experiment> {
}
