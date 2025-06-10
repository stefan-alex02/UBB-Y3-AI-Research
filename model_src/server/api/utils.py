from datetime import datetime

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Union, Optional


# --- Experiment Related ---
class ExperimentMethodParams(BaseModel):
    method_name: str = Field(..., description="Name of the pipeline method, e.g., 'single_train'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the method's 'params' arg (e.g., Skorch HPs or GridSearchCV config)")
    param_grid: Optional[Dict[str, Any]] = None

    save_model: Optional[bool] = None
    save_best_model: Optional[bool] = None
    plot_level: Optional[int] = None
    results_detail_level: Optional[int] = None
    cv: Optional[int] = None
    outer_cv: Optional[int] = None
    inner_cv: Optional[int] = None
    scoring: Optional[str] = None
    method_search_type: Optional[str] = Field(None, description="'grid' or 'random' for search methods")
    n_iter: Optional[int] = None
    evaluate_on: Optional[str] = None
    val_split_ratio: Optional[float] = None
    use_best_params_from_step: Optional[int] = None

    class Config:
        pass # Keep if needed for other Pydantic settings


class RunExperimentRequest(BaseModel): # This is what Python's /experiments/run endpoint expects
    experiment_run_id: str = Field(..., description="System-generated unique ID from Java, used for artifact paths")
    dataset_name: str
    model_type: str
    methods_sequence: List[ExperimentMethodParams] # List of the Pydantic model above
    img_size_h: Optional[int] = None
    img_size_w: Optional[int] = None
    offline_augmentation: Optional[bool] = False
    augmentation_strategy_override: Optional[str] = None
    test_split_ratio_if_flat: Optional[float] = None
    random_seed: Optional[int] = None
    force_flat_for_fixed_cv: Optional[bool] = False
    # save_model_default: Optional[bool] = None # If you removed this global flag


class ExperimentRunResponse(BaseModel):
    experiment_run_id: str
    message: str
    status: str = "SUBMITTED"


class ArtifactNode(BaseModel):
    name: str
    path: str # Full path relative to experiment_run_id folder for files, or relative to prefix for folders
    type: str # 'file' or 'folder' or 'log', 'json', 'csv', 'png', 'pt' etc.
    children: Optional[List['ArtifactNode']] = None
    size: Optional[int] = None # Optional: file size in bytes
    last_modified: Optional[datetime] = None # Optional: last modified timestamp
ArtifactNode.model_rebuild()


# --- Prediction Related ---
class ImageIdFormatPair(BaseModel):
    image_id: Union[int, str] # SQL PK for the image
    image_format: str # e.g. "png", "jpg"


class ModelLoadDetails(BaseModel):
    # Information needed to reconstruct the model path for loading
    # These parts form the prefix: experiments/{dataset_name_of_model}/{model_type_of_model}/{experiment_run_id_of_model_producer}/
    dataset_name_of_model: str
    model_type_of_model: str
    experiment_run_id_of_model_producer: str
    # This is the path relative to the prefix above
    relative_model_path_in_experiment: str # e.g., "single_train_0/cnn_epoch5_val_loss0.123.pt"


class RunPredictionRequest(BaseModel):
    username: str
    image_id_format_pairs: List[ImageIdFormatPair]
    model_load_details: ModelLoadDetails
    experiment_run_id_of_model: str # ID of the experiment that *produced* the model (for result grouping)
    generate_lime: Optional[bool] = False
    lime_num_features: Optional[int] = 5
    lime_num_samples: Optional[int] = 100
    prob_plot_top_k: Optional[int] = -1


class SinglePredictionResult(BaseModel):
    image_id: Union[int, str]
    experiment_id: str # experiment_run_id_of_model
    predicted_class: str
    confidence: float


class PredictionRunResponse(BaseModel):
    predictions: List[SinglePredictionResult]
    message: str
