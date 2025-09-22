from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Union


class ArtifactRepository(ABC):
    @abstractmethod
    def save_json(self, data: Dict, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def save_model_state_dict(self, state_dict: Dict, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def save_plot_figure(self, fig, key: str) -> Optional[str]: # fig is matplotlib figure
        pass

    @abstractmethod
    def save_text_file(self, content: str, key: str, content_type: str = 'text/plain') -> Optional[str]:
        pass

    @abstractmethod
    def load_json(self, key: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def load_model_state_dict(self, key: str, map_location: Optional[str] = None) -> Optional[Dict]:
        pass

    @abstractmethod
    def get_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        pass

    @abstractmethod
    def upload_file(self, local_file_path: Union[str, Path], key: str) -> Optional[str]:
        """Uploads a local file to the repository. Returns identifier (path/key) or None."""
        pass

    @abstractmethod
    def save_image_object(self, image_bytes: bytes, key: str, content_type: str = 'image/png') -> Optional[str]:
        pass

    @abstractmethod
    def delete_object(self, key: str) -> bool:
        """Deletes a single object. Returns True on success, False otherwise."""
        pass

    @abstractmethod
    def delete_objects_by_prefix(self, prefix: str) -> bool:
        """Deates all objects under a given prefix. Returns True if all deletions were attempted (some might fail silently depending on client)."""
        pass
