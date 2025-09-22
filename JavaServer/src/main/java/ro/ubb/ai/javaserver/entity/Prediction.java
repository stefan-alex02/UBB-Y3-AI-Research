package ro.ubb.ai.javaserver.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "Predictions")
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "image_id", nullable = false)
    private Image image;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "experiment_run_id_of_model", nullable = true)
    private Experiment modelExperiment;

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