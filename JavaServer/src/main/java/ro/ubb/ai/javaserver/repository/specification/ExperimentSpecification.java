package ro.ubb.ai.javaserver.repository.specification;

import ro.ubb.ai.javaserver.dto.experiment.ExperimentFilterDTO;
import ro.ubb.ai.javaserver.entity.Experiment;
import jakarta.persistence.criteria.Predicate;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;

@Component
public class ExperimentSpecification {

    public static final String NAME = "name"; // User-defined name
    public static final String MODEL_TYPE = "modelType";
    public static final String DATASET_NAME = "datasetName";
    public static final String STATUS = "status";
    public static final String START_TIME = "startTime";
    public static final String END_TIME = "endTime";
    public static final String MODEL_RELATIVE_PATH = "modelRelativePath";

    public static Specification<Experiment> fromFilter(ExperimentFilterDTO filter) {
        return (root, query, cb) -> { // cb for criteriaBuilder
            List<Predicate> predicates = new ArrayList<>();

            if (filter == null) {
                return cb.conjunction();
            }

            if (StringUtils.hasText(filter.getNameContains())) {
                predicates.add(cb.like(
                        cb.lower(root.get(NAME)), // User-defined name
                        "%" + filter.getNameContains().toLowerCase() + "%"
                ));
            }
            if (StringUtils.hasText(filter.getModelType())) {
                predicates.add(cb.equal(root.get(MODEL_TYPE), filter.getModelType()));
            }
            if (StringUtils.hasText(filter.getDatasetName())) {
                predicates.add(cb.equal(root.get(DATASET_NAME), filter.getDatasetName()));
            }
            if (filter.getStatus() != null) {
                predicates.add(cb.equal(root.get(STATUS), filter.getStatus()));
            }
            if (filter.getHasModelSaved() != null) {
                if (Boolean.TRUE.equals(filter.getHasModelSaved())) {
                    predicates.add(cb.isNotNull(root.get(MODEL_RELATIVE_PATH)));
                    predicates.add(cb.notEqual(root.get(MODEL_RELATIVE_PATH), "")); // Also check not empty string
                } else { // hasModelSaved is false
                    predicates.add(cb.or(
                            cb.isNull(root.get(MODEL_RELATIVE_PATH)),
                            cb.equal(root.get(MODEL_RELATIVE_PATH), "")
                    ));
                }
            }
            if (filter.getStartedAfter() != null) {
                predicates.add(cb.greaterThanOrEqualTo(root.get(START_TIME), filter.getStartedAfter()));
            }
            if (filter.getFinishedBefore() != null) {
                predicates.add(cb.isNotNull(root.get(END_TIME))); // Only consider if ended
                predicates.add(cb.lessThanOrEqualTo(root.get(END_TIME), filter.getFinishedBefore()));
            }

            return cb.and(predicates.toArray(new Predicate[0]));
        };
    }
}