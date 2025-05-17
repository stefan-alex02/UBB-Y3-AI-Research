import json
import logging
import shutil
from datetime import datetime  # For JSON serializer
from pathlib import Path
from typing import Optional, Dict, Union, Callable  # Add List if not already there

import numpy as np  # For JSON serializer
import torch  # For saving/loading model state_dict
from skorch.callbacks import Callback
from skorch.dataset import ValidSplit
from torch import nn

from .artifact_repo import ArtifactRepository

logger = logging.getLogger(__name__)


class LocalFileSystemRepository(ArtifactRepository):
    def __init__(self, base_storage_path: Union[str, Path]):
        self.base_path = Path(base_storage_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalFileSystemRepository initialized at: {self.base_path}")

    def _get_full_path(self, key: str) -> Path:
        return self.base_path / key

    def _json_serializer(self, obj): # Moved serializer to a helper
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, Path): return str(obj)
        elif isinstance(obj, datetime): return obj.isoformat()
        elif isinstance(obj, (slice, type, Callable)): return None # Added Callable
        elif isinstance(obj, (torch.optim.Optimizer, nn.Module, Callback)): return str(type(obj).__name__)
        elif isinstance(obj, ValidSplit): return f"ValidSplit(cv={obj.cv}, stratified={obj.stratified})"
        try: return json.JSONEncoder.default(self, obj) # Pass self to default for consistency
        except TypeError: return str(obj)

    def save_json(self, data: Dict, key: str) -> Optional[str]:
        full_path = self._get_full_path(key)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, default=self._json_serializer)
            logger.info(f"JSON saved locally: {full_path}")
            return str(full_path)
        except Exception as e: logger.error(f"Failed to save JSON locally to {full_path}: {e}"); return None

    def save_model_state_dict(self, state_dict: Dict, key: str) -> Optional[str]:
        full_path = self._get_full_path(key)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, full_path)
            logger.info(f"Model state_dict saved locally: {full_path}")
            return str(full_path)
        except Exception as e: logger.error(f"Failed to save model state_dict locally to {full_path}: {e}"); return None

    def save_plot_figure(self, fig, key: str) -> Optional[str]: # fig from matplotlib
        full_path = self._get_full_path(key)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(pad=1.5)
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved locally: {full_path}")
            return str(full_path)
        except Exception as e: logger.error(f"Failed to save plot locally to {full_path}: {e}"); return None

    def save_text_file(self, content: str, key: str) -> Optional[str]:
        full_path = self._get_full_path(key)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f: f.write(content)
            logger.info(f"Text file saved locally: {full_path}")
            return str(full_path)
        except Exception as e: logger.error(f"Failed to save text file locally to {full_path}: {e}"); return None

    def load_json(self, key: str) -> Optional[Dict]:
        full_path = self._get_full_path(key)
        if not full_path.is_file(): logger.error(f"JSON file not found locally: {full_path}"); return None
        try:
            with open(full_path, 'r', encoding='utf-8') as f: data = json.load(f)
            logger.info(f"JSON loaded locally from: {full_path}")
            return data
        except Exception as e: logger.error(f"Failed to load JSON locally from {full_path}: {e}"); return None

    def load_model_state_dict(self, key: str, map_location: Optional[str] = None) -> Optional[Dict]:
        full_path = self._get_full_path(key)
        if not full_path.is_file(): logger.error(f"Model state_dict file not found locally: {full_path}"); return None
        try:
            state_dict = torch.load(full_path, map_location=map_location, weights_only=True)
            logger.info(f"Model state_dict loaded locally from: {full_path}")
            return state_dict
        except Exception as e: logger.error(f"Failed to load model state_dict locally from {full_path}: {e}", exc_info=True); return None

    def get_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        logger.warning("Presigned URLs are not applicable for LocalFileSystemRepository. Returning local file path.")
        local_path = self._get_full_path(object_key)
        if local_path.exists():
            return local_path.as_uri() # file:///...
        return None

    def upload_file(self, local_file_path: Union[str, Path], key: str) -> Optional[str]:
        source_path = Path(local_file_path)
        destination_path = self._get_full_path(key)

        if not source_path.is_file():
            logger.error(f"Local source file for upload not found: {source_path}")
            return None
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
            logger.info(f"File copied locally from {source_path} to {destination_path}")
            return str(destination_path)
        except Exception as e:
            logger.error(f"Failed to copy file locally from {source_path} to {destination_path}: {e}")
            return None
