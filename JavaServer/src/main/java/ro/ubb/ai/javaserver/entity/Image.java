package ro.ubb.ai.javaserver.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "Images")
public class Image {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(length = 10, nullable = false)
    private String format;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "uploaded_at", columnDefinition = "TIMESTAMP WITH TIME ZONE")
    private OffsetDateTime uploadedAt;

    @OneToMany(mappedBy = "image", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Prediction> predictions;

    @PrePersist
    protected void onUpload() {
        if (uploadedAt == null) {
            uploadedAt = OffsetDateTime.now();
        }
    }
}