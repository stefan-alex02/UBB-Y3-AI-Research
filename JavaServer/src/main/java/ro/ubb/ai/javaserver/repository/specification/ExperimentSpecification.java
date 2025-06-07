package ro.ubb.ai.javaserver.repository.specification;

import ro.ubb.ai.javaserver.dto.experiment.ExperimentFilterDTO;
import ro.ubb.ai.javaserver.entity.Experiment;
import jakarta.persistence.criteria.Predicate;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils; // For StringUtils.hasText

import java.util.ArrayList;
import java.util.List;

@Component // Make it a Spring component if you want to autowire it, though static methods are also fine
public class ExperimentSpecification {

    public static Specification<Experiment> fromFilter(ExperimentFilterDTO filter) {
        return (root, query, criteriaBuilder) -> {
            List<Predicate> predicates = new ArrayList<>();

            if (filter == null) {
                return criteriaBuilder.conjunction(); // No filters, return all
            }

            if (StringUtils.hasText(filter.getModelType())) {
                predicates.add(criteriaBuilder.equal(
                        criteriaBuilder.lower(root.get("modelType")), // Use metamodel for type safety
                        filter.getModelType().toLowerCase()
                ));
            }
            if (StringUtils.hasText(filter.getDatasetName())) {
                predicates.add(criteriaBuilder.equal(
                        criteriaBuilder.lower(root.get("datasetName")),
                        filter.getDatasetName().toLowerCase()
                ));
            }
            if (filter.getStatus() != null) {
                predicates.add(criteriaBuilder.equal(root.get("status"), filter.getStatus()));
            }
            if (filter.getStartedAfter() != null) {
                predicates.add(criteriaBuilder.greaterThanOrEqualTo(root.get("startTime"), filter.getStartedAfter()));
            }
            if (filter.getFinishedBefore() != null) {
                // If filtering for finished experiments, ensure endTime is not null
                predicates.add(criteriaBuilder.isNotNull(root.get("endTime")));
                predicates.add(criteriaBuilder.lessThanOrEqualTo(root.get("endTime"), filter.getFinishedBefore()));
            }
            // Add more filters here, e.g., by name (LIKE query), by user_id etc.

            return criteriaBuilder.and(predicates.toArray(new Predicate[0]));
        };
    }
}
