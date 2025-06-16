from datetime import datetime
from typing import List, Dict, Any, Union, Optional

from pydantic import BaseModel, Field


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
        pass


class RunExperimentRequest(BaseModel):
    experiment_run_id: str = Field(..., description="System-generated unique ID from Java, used for artifact paths")
    dataset_name: str
    model_type: str
    methods_sequence: List[ExperimentMethodParams]
    img_size_h: Optional[int] = None
    img_size_w: Optional[int] = None
    offline_augmentation: Optional[bool] = False
    augmentation_strategy_override: Optional[str] = None
    test_split_ratio_if_flat: Optional[float] = None
    random_seed: Optional[int] = None
    force_flat_for_fixed_cv: Optional[bool] = False


class ExperimentRunResponse(BaseModel):
    experiment_run_id: str
    message: str
    status: str = "SUBMITTED"


class ArtifactNode(BaseModel):
    name: str
    path: str
    type: str
    children: Optional[List['ArtifactNode']] = None
    size: Optional[int] = None
    last_modified: Optional[datetime] = None
ArtifactNode.model_rebuild()


class ImagePredictionTask(BaseModel):
    image_id: str
    image_format: str
    prediction_id: str


class ModelLoadDetails(BaseModel):
    dataset_name_of_model: str
    model_type_of_model: str
    experiment_run_id_of_model_producer: str
    relative_model_path_in_experiment: str


class RunPredictionRequest(BaseModel):
    username: str
    image_prediction_tasks: List[ImagePredictionTask]
    model_load_details: Optional[ModelLoadDetails]
    experiment_run_id_of_model: Optional[str] = None

    generate_lime: Optional[bool] = False
    lime_num_features: Optional[int] = 5
    lime_num_samples: Optional[int] = 100
    prob_plot_top_k: Optional[int] = -1


class SinglePredictionResult(BaseModel):
    prediction_id: str
    image_id: Union[int, str]
    experiment_id: str
    predicted_class: str
    confidence: float


class PredictionRunResponse(BaseModel):
    predictions: List[SinglePredictionResult]
    message: str
