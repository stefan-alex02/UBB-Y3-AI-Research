package ro.ubb.ai.javaserver.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import ro.ubb.ai.javaserver.enums.ExperimentStatus;

import java.time.OffsetDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "Experiments")
public class Experiment {

    @Id
    @Column(name = "experiment_run_id", length = 255)
    private String experimentRunId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false)
    private String name;

    @Column(name = "model_type", length = 50, nullable = false)
    private String modelType;

    @Column(name = "dataset_name", length = 100, nullable = false)
    private String datasetName;

    @Column(name = "model_relative_path", length = 512)
    private String modelRelativePath;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private ExperimentStatus status = ExperimentStatus.PENDING;

    @Column(name = "start_time", columnDefinition = "TIMESTAMP WITH TIME ZONE")
    private OffsetDateTime startTime;

    @Column(name = "end_time", columnDefinition = "TIMESTAMP WITH TIME ZONE")
    private OffsetDateTime endTime;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "sequence_config", columnDefinition = "jsonb")
    private String sequenceConfig;

    @PrePersist
    protected void onCreate() {
        if (startTime == null) {
            startTime = OffsetDateTime.now();
        }
    }
}