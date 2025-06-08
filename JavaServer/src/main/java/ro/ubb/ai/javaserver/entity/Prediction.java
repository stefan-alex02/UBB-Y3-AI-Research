package ro.ubb.ai.javaserver.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import java.time.OffsetDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "Predictions"
        // The unique constraint in SQL will handle the NULLs correctly.
        // JPA/Hibernate might require specific handling if you were defining it via annotations
        // for schema generation with NULLs, but since you manage schema with SQL, this is fine.
        // uniqueConstraints = {
        //    @UniqueConstraint(columnNames = {"image_id", "experiment_run_id_of_model"})
        // }
)
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "image_id", nullable = false)
    private Image image;

    @ManyToOne(fetch = FetchType.LAZY) // An experiment that produced the model
    @JoinColumn(name = "experiment_run_id_of_model", nullable = true) // <<<< ALLOW NULLABLE JOIN COLUMN
    private Experiment modelExperiment; // Renamed for clarity

    @Column(name = "predicted_class", length = 255, nullable = false)
    private String predictedClass;

    @Column(nullable = false)
    private Float confidence;

    @Column(name = "prediction_timestamp", columnDefinition = "TIMESTAMP WITH TIME ZONE")
    private OffsetDateTime predictionTimestamp;

    @PrePersist
    protected void onPrediction() {
        if (predictionTimestamp == null) {
            predictionTimestamp = OffsetDateTime.now();
        }
    }
}