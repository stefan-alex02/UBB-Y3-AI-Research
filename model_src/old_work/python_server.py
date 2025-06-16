# python_server.py
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import torch # For torch.no_grad

from model_src.server.ml.pipeline import ClassificationPipeline
from model_src.server.ml import RANDOM_SEED # for LIME
from model_src.server.ml import ModelType
from model_src.server.ml import logger, setup_logger, logger_name_global

# Assuming LIME and skimage are installed
try:
    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import mark_boundaries
    LIME_SERVER_AVAILABLE = True
except ImportError:
    LIME_SERVER_AVAILABLE = False
    LimeImageExplainer = None
    mark_boundaries = None
    logger.warning("LIME or scikit-image not found. LIME features will be disabled on the server.")


# --- Global Variables & Setup ---
app = Flask(__name__)
CORS(app)


# ... (PIPELINE_INSTANCE, MODEL_LOADED_SUCCESSFULLY, LIME_EXPLAINER_INSTANCE setup - unchanged) ...
PIPELINE_INSTANCE: Optional[ClassificationPipeline] = None
MODEL_LOADED_SUCCESSFULLY = False
LIME_EXPLAINER_INSTANCE: Optional[LimeImageExplainer] = None

def get_lime_explainer():
    global LIME_EXPLAINER_INSTANCE
    if LIME_EXPLAINER_INSTANCE is None and LIME_SERVER_AVAILABLE:
        LIME_EXPLAINER_INSTANCE = LimeImageExplainer(random_state=RANDOM_SEED) # Use your RANDOM_SEED
    return LIME_EXPLAINER_INSTANCE

def server_lime_predict_fn(numpy_images_batch_lime: np.ndarray) -> np.ndarray:
    if PIPELINE_INSTANCE is None or not PIPELINE_INSTANCE.model_adapter.initialized_: raise RuntimeError("LIME predict_fn: Pipeline or model not ready.")
    processed_images_lime = []
    for img_np_lime in numpy_images_batch_lime:
        if img_np_lime.dtype == np.double or img_np_lime.dtype == np.float64 or img_np_lime.dtype == np.float32:
             if img_np_lime.max() <= 1.0 and img_np_lime.min() >=0.0: img_np_lime = (img_np_lime * 255).astype(np.uint8)
             else: img_np_lime = np.clip(img_np_lime, 0, 255).astype(np.uint8)
        elif img_np_lime.dtype != np.uint8: img_np_lime = np.clip(img_np_lime, 0, 255).astype(np.uint8)
        pil_img_lime = Image.fromarray(img_np_lime); transformed_img_lime = PIPELINE_INSTANCE.dataset_handler.get_eval_transform()(pil_img_lime); processed_images_lime.append(transformed_img_lime)
    if not processed_images_lime: return np.array([])
    batch_tensor_lime = torch.stack(processed_images_lime).to(PIPELINE_INSTANCE.model_adapter.device); PIPELINE_INSTANCE.model_adapter.module_.eval()
    with torch.no_grad(): logits_lime = PIPELINE_INSTANCE.model_adapter.module_(batch_tensor_lime); probs_lime = torch.softmax(logits_lime, dim=1)
    return probs_lime.cpu().numpy()


def initialize_pipeline_server():
    global PIPELINE_INSTANCE, MODEL_LOADED_SUCCESSFULLY
    DEFAULT_MODEL_TYPE_STR = os.environ.get("DEFAULT_MODEL_TYPE", "cnn")
    # DEFAULT_MODEL_PATH = os.environ.get("DEFAULT_MODEL_PATH", None)
    DEFAULT_MODEL_PATH = os.environ.get("DEFAULT_MODEL_PATH", "./results/Swimcat-extend/cnn/20250515_160130_seed42/single_train_20250515_160130_450999/cnn_epoch4_val_valid-loss0.3059.pt")
    DEFAULT_DATASET_FOR_HANDLER = os.environ.get("DEFAULT_DATASET_PATH", "../data/Swimcat-extend")
    logger.info(f"Attempting to initialize pipeline with model_type='{DEFAULT_MODEL_TYPE_STR}', model_path='{DEFAULT_MODEL_PATH}'")
    try:
        if not Path(DEFAULT_DATASET_FOR_HANDLER).exists(): logger.error(f"Reference dataset path for handler ('{DEFAULT_DATASET_FOR_HANDLER}') not found. Cannot initialize pipeline."); MODEL_LOADED_SUCCESSFULLY = False; PIPELINE_INSTANCE = None; return
        try: mt_enum = ModelType(DEFAULT_MODEL_TYPE_STR.lower())
        except ValueError: logger.error(f"Invalid model_type string: '{DEFAULT_MODEL_TYPE_STR}'"); MODEL_LOADED_SUCCESSFULLY = False; PIPELINE_INSTANCE = None; return
        PIPELINE_INSTANCE = ClassificationPipeline(dataset_path=DEFAULT_DATASET_FOR_HANDLER, model_type=mt_enum, model_load_path=DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH and Path(DEFAULT_MODEL_PATH).exists() else None, results_dir=None, plot_level=0, results_detail_level=0)
        if PIPELINE_INSTANCE.model_adapter.initialized_: logger.info(f"Pipeline initialized successfully with model: {DEFAULT_MODEL_TYPE_STR}"); MODEL_LOADED_SUCCESSFULLY = True
        else: logger.error(f"Pipeline init OK BUT model adapter not ready for: {DEFAULT_MODEL_TYPE_STR}"); MODEL_LOADED_SUCCESSFULLY = False; PIPELINE_INSTANCE = None
    except Exception as e: logger.error(f"Failed to initialize ClassificationPipeline: {e}", exc_info=True); PIPELINE_INSTANCE = None; MODEL_LOADED_SUCCESSFULLY = False


if not logger.hasHandlers() or all(isinstance(h, logging.NullHandler) for h in logger.handlers):
    setup_logger(name=logger_name_global, log_dir=None, level=logging.INFO)
initialize_pipeline_server()


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if not MODEL_LOADED_SUCCESSFULLY or PIPELINE_INSTANCE is None: return jsonify({"error": "Model service not ready."}), 503
    pil_images_loaded: List[Tuple[str, Optional[Image.Image]]] = []
    if 'images' in request.files:
        uploaded_files = request.files.getlist('images')
        for file_storage in uploaded_files:
            if file_storage and file_storage.filename:
                try: image_bytes = file_storage.read(); img = Image.open(io.BytesIO(image_bytes)).convert('RGB'); pil_images_loaded.append((file_storage.filename, img))
                except Exception as e: logger.warning(f"Could not process uploaded file {file_storage.filename}: {e}"); pil_images_loaded.append((file_storage.filename, None))
            else: logger.warning("Received an empty file part in 'images'.")
    elif request.is_json and 'image_data_list' in request.json:
        image_data_list = request.json['image_data_list']
        if not isinstance(image_data_list, list): return jsonify({"error": "'image_data_list' must be a list."}), 400
        for item_idx, item_data in enumerate(image_data_list):
            if not isinstance(item_data, dict): logger.warning(f"Item {item_idx} in 'image_data_list' not dict. Skipping."); continue
            source_type = item_data.get('type'); data = item_data.get('data'); identifier = item_data.get('identifier', f"json_source_{item_idx}"); pil_img_json = None
            if not source_type or not data: logger.warning(f"Item {item_idx} missing type/data. Skipping {identifier}."); continue
            try:
                if source_type == 'base64':
                    if ',' in data: data = data.split(',', 1)[1]
                    image_bytes = base64.b64decode(data); pil_img_json = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                elif source_type == 'url':
                    response = requests.get(data, timeout=10); response.raise_for_status(); pil_img_json = Image.open(io.BytesIO(response.content)).convert('RGB')
                else: logger.warning(f"Unsupported type '{source_type}' for {identifier}.")
            except Exception as e: logger.warning(f"Could not process JSON image source {identifier}: {e}")
            pil_images_loaded.append((identifier, pil_img_json))
    else: return jsonify({"error": "No 'images' (form-data) or 'image_data_list' (JSON) in request."}), 400
    if not pil_images_loaded: return jsonify({"error": "No valid images processed from request."}), 400
    image_sources_for_pipeline = [item[1] for item in pil_images_loaded if item[1] is not None]; identifiers_for_pipeline = [item[0] for item in pil_images_loaded if item[1] is not None]; pil_map_for_lime_plot = {ident:img for ident, img in pil_images_loaded if img is not None}
    if not image_sources_for_pipeline: return jsonify({"error": "All images failed initial loading/parsing."}), 400
    generate_lime = request.args.get('lime', 'false').lower() == 'true'; lime_features = int(request.args.get('lime_features', 5)); lime_samples = int(request.args.get('lime_samples', 100))
    try:
        prediction_results = PIPELINE_INSTANCE.predict_images(image_sources=image_sources_for_pipeline,
                                                              original_identifiers=identifiers_for_pipeline,
                                                              results_detail_level=0, plot_level=0,
                                                              generate_lime_explanations=generate_lime,
                                                              lime_num_features=lime_features,
                                                              lime_num_samples=lime_samples)
        if generate_lime and LIME_SERVER_AVAILABLE and mark_boundaries:
            lime_explainer_for_server = get_lime_explainer()
            for pred_item in prediction_results:
                lime_data = pred_item.get('lime_explanation'); identifier = pred_item.get('identifier'); pil_original_image = pil_map_for_lime_plot.get(identifier)
                if lime_data and isinstance(lime_data, dict) and not lime_data.get('error') and pil_original_image and lime_explainer_for_server:
                    try:
                        img_np_original = np.array(pil_original_image)
                        explanation_obj = lime_explainer_for_server.explain_instance(image=img_np_original, classifier_fn=server_lime_predict_fn, top_labels=1, hide_color=0, num_features=lime_features, num_samples=lime_samples, random_seed=42)
                        temp, mask = explanation_obj.get_image_and_mask(label=pred_item['predicted_class_idx'], positive_only=True, num_features=lime_features, hide_rest=False)
                        lime_image_np = mark_boundaries(temp / 2 + 0.5 if temp.max() > 1.0 else temp, mask, color=(1,0,0), mode='thick', outline_color=(1,0,0)); lime_image_pil = Image.fromarray((lime_image_np * 255).astype(np.uint8))
                        buffered = io.BytesIO(); lime_image_pil.save(buffered, format="PNG"); lime_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8'); pred_item['lime_explanation']['lime_image_base64'] = lime_img_str
                    except Exception as e_lime_render: logger.error(f"Error rendering LIME image for {identifier}: {e_lime_render}"); pred_item['lime_explanation']['lime_image_base64_error'] = str(e_lime_render)
        return jsonify({"predictions": prediction_results}), 200
    except Exception as e: logger.error(f"Error in /predict endpoint: {e}", exc_info=True); return jsonify({"error": "Prediction failed due to an internal server error."}), 500


if __name__ == '__main__':
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5001))
    logger.info(f"Starting Flask server on {host}:{port}...")
    app.run(host=host, port=port, debug=False) # Set debug=True for development ONLY
