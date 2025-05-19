import logging
import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

from .artifact_repo import ArtifactRepository
from .file_repo import LocalFileSystemRepository
from .minio_repo import MinIORepository


def load_minio_repository(logger: logging.Logger,
                          bucket_name: str = "my-ml-experiments"
                          ) -> Optional[MinIORepository]:
    try:
        load_dotenv()
        # These could also come from a config file or CLI args
        repo: Optional[ArtifactRepository] = MinIORepository(
            bucket_name=bucket_name,
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", "http://127.0.0.1:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "N/A"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "N/A"),
            # region_name="us-east-1" # Often optional for MinIO
        )
        logger.info(f"Using MinIORepository, targeting bucket: {bucket_name}")
    except Exception as e:
        logger.error(f"MinIO Repository connection failed: {e}. File outputs will be disabled.")
        repo = None

    return repo


def load_file_repository(logger: logging.Logger,
                         repo_base_path: Union[str, Path]
                         ) -> Optional[LocalFileSystemRepository]:
    try:
        repo: Optional[ArtifactRepository] = LocalFileSystemRepository(base_storage_path=repo_base_path)
        logger.info(f"Using LocalFileSystemRepository, base path: {repo_base_path}")
    except Exception as e:
        logger.error(f"LocalFileSystemRepository setup failed: {e}. File outputs might be disabled.")
        repo = None

    return repo
