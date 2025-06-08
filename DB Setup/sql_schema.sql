CREATE TABLE Users (
    id BIGSERIAL PRIMARY KEY, -- <--- CHANGE TO BIGSERIAL
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    role VARCHAR(20) NOT NULL CHECK (role IN ('NORMAL', 'METEOROLOGIST')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Images Table
CREATE TABLE Images (
    id BIGSERIAL PRIMARY KEY, -- <--- CHANGE TO BIGSERIAL
    format VARCHAR(10) NOT NULL,
    user_id BIGINT NOT NULL, -- <--- Must match User.id type
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE
);

-- Experiments Table
CREATE TABLE Experiments (
    experiment_run_id VARCHAR(255) PRIMARY KEY,
    user_id BIGINT NOT NULL, -- <--- Must match User.id type
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    model_relative_path VARCHAR(512),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    sequence_config JSONB,
    FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE SET NULL
);

-- Predictions Table
CREATE TABLE Predictions (
    id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL,
    experiment_run_id_of_model VARCHAR(255) NULL,
    predicted_class VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	
	UNIQUE (image_id, experiment_run_id_of_model), -- This still works, NULLs are not equal.
    FOREIGN KEY (image_id) REFERENCES Images(id) ON DELETE CASCADE,
    FOREIGN KEY (experiment_run_id_of_model) REFERENCES Experiments(experiment_run_id) ON DELETE SET NULL
);
CREATE INDEX idx_predictions_image_id ON Predictions(image_id);
CREATE INDEX idx_predictions_experiment_id ON Predictions(experiment_run_id_of_model);