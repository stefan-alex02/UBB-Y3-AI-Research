import io
import json
import logging
import os
from datetime import datetime  # For JSON serializer
from pathlib import Path
from typing import Optional, Dict, Union, Callable  # Add List if not already there

import boto3
import numpy as np  # For JSON serializer
import torch  # For saving/loading model state_dict
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from skorch.callbacks import Callback
from skorch.dataset import ValidSplit
from torch import nn

from .artifact_repo import ArtifactRepository

logger = logging.getLogger(__name__)


class MinIORepository(ArtifactRepository):
    def __init__(self, bucket_name: str,
                 endpoint_url: Optional[str] = None, access_key: Optional[str] = None,
                 secret_key: Optional[str] = None, region_name: Optional[str] = None):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url or os.environ.get("MINIO_ENDPOINT_URL", "http://127.0.0.1:9000")
        self.access_key = access_key or os.environ.get("MINIO_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("MINIO_SECRET_KEY")
        self.region_name = region_name or os.environ.get("MINIO_REGION") # Can be None
        self.secure = self.endpoint_url.startswith("https://")

        if not self.access_key or not self.secret_key:
            logger.error("MinIO access key or secret key not provided or found in environment variables.")
            raise ValueError("MinIO credentials missing.")
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name,
                use_ssl=self.secure,
                config=boto3.session.Config(s3={'addressing_style': 'path'}) # Often needed for MinIO
            )
            self.client.list_buckets() # Test connection
            logger.info(f"Successfully connected to MinIO at {self.endpoint_url}, bucket '{self.bucket_name}' will be used.")
            self._ensure_bucket_exists(self.bucket_name)
        except (NoCredentialsError, PartialCredentialsError) as e: logger.error(f"MinIO credentials error: {e}"); raise
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == 'InvalidAccessKeyId' or error_code == 'SignatureDoesNotMatch': logger.error(f"MinIO authentication failed: {e}")
            else: logger.error(f"Failed to connect to MinIO or list buckets: {e}")
            raise
        except Exception as e: logger.error(f"Unexpected error during MinIO client init: {e}"); raise

    def _ensure_bucket_exists(self, bucket_name: str):
        try: self.client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == '404' or error_code == 'NoSuchBucket':
                try: self.client.create_bucket(Bucket=bucket_name); logger.info(f"Bucket '{bucket_name}' created.")
                except Exception as ce: logger.error(f"Failed to create bucket '{bucket_name}': {ce}"); raise
            else: logger.error(f"Error checking bucket '{bucket_name}': {e}"); raise

    def _json_serializer(self, obj): # Moved serializer to a helper
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, Path): return str(obj)
        elif isinstance(obj, datetime): return obj.isoformat()
        elif isinstance(obj, (slice, type, Callable)): return None
        elif isinstance(obj, (torch.optim.Optimizer, nn.Module, Callback)): return str(type(obj).__name__)
        elif isinstance(obj, ValidSplit): return f"ValidSplit(cv={obj.cv}, stratified={obj.stratified})"
        try: return json.JSONEncoder.default(self, obj)
        except TypeError: return str(obj)

    def save_json(self, data: Dict, key: str) -> Optional[str]:
        try:
            json_string = json.dumps(data, indent=4, default=self._json_serializer)
            self.client.put_object(Bucket=self.bucket_name, Key=key, Body=json_string.encode('utf-8'), ContentType='application/json')
            logger.info(f"JSON saved to S3: s3://{self.bucket_name}/{key}")
            return f"s3://{self.bucket_name}/{key}"
        except Exception as e: logger.error(f"Failed to save JSON to S3 (s3://{self.bucket_name}/{key}): {e}"); return None

    def save_model_state_dict(self, state_dict: Dict, key: str) -> Optional[str]:
        try:
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            buffer.seek(0)
            self.client.put_object(Bucket=self.bucket_name, Key=key, Body=buffer.getvalue(), ContentType='application/octet-stream')
            logger.info(f"Model state_dict saved to S3: s3://{self.bucket_name}/{key}")
            return f"s3://{self.bucket_name}/{key}"
        except Exception as e: logger.error(f"Failed to save model state_dict to S3 (s3://{self.bucket_name}/{key}): {e}"); return None

    def save_plot_figure(self, fig, key: str) -> Optional[str]: # fig from matplotlib
        try:
            img_buffer = io.BytesIO()
            fig.tight_layout(pad=1.5)
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            self.client.put_object(Bucket=self.bucket_name, Key=key, Body=img_buffer.getvalue(), ContentType='image/png')
            logger.info(f"Plot saved to S3: s3://{self.bucket_name}/{key}")
            return f"s3://{self.bucket_name}/{key}"
        except Exception as e: logger.error(f"Failed to save plot to S3 (s3://{self.bucket_name}/{key}): {e}"); return None

    def save_text_file(self, content: str, key: str) -> Optional[str]:
        try:
            self.client.put_object(Bucket=self.bucket_name, Key=key, Body=content.encode('utf-8'), ContentType='text/plain')
            logger.info(f"Text file saved to S3: s3://{self.bucket_name}/{key}")
            return f"s3://{self.bucket_name}/{key}"
        except Exception as e: logger.error(f"Failed to save text file to S3 (s3://{self.bucket_name}/{key}): {e}"); return None

    def download_file_to_memory(self, object_key: str) -> Optional[bytes]: # bucket_name is instance member
        """Downloads a file from MinIO into memory (bytes)."""
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
            file_content = response['Body'].read()
            logger.debug(f"Successfully downloaded {object_key} from {self.bucket_name} to memory.")
            return file_content
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey': logger.error(f"File not found in MinIO: s3://{self.bucket_name}/{object_key}")
            else: logger.error(f"Error downloading {object_key} from s3://{self.bucket_name}: {e}")
        except Exception as e: logger.error(f"Unexpected error downloading {object_key} from s3://{self.bucket_name}: {e}")
        return None

    def load_json(self, key: str) -> Optional[Dict]:
        try:
            json_bytes = self.download_file_to_memory(key)
            if json_bytes:
                data = json.loads(json_bytes.decode('utf-8'))
                logger.info(f"JSON loaded from S3: s3://{self.bucket_name}/{key}")
                return data
        except Exception as e: logger.error(f"Failed to parse JSON from S3 (s3://{self.bucket_name}/{key}): {e}")
        return None

    def load_model_state_dict(self, key: str, map_location: Optional[str] = None) -> Optional[Dict]:
        try:
            model_bytes = self.download_file_to_memory(key)
            if model_bytes:
                buffer = io.BytesIO(model_bytes)
                state_dict = torch.load(buffer, map_location=map_location, weights_only=True)
                logger.info(f"Model state_dict loaded from S3: s3://{self.bucket_name}/{key}")
                return state_dict
        except Exception as e: logger.error(f"Failed to load model state_dict from S3 (s3://{self.bucket_name}/{key}): {e}", exc_info=True)
        return None

    def get_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        try:
            response = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return response
        except Exception as e: logger.error(f"Failed to generate presigned URL for s3://{self.bucket_name}/{object_key}: {e}"); return None

    def upload_file(self, local_file_path: Union[str, Path], key: str) -> Optional[str]:
        """Uploads a local file to MinIO."""
        self._ensure_bucket_exists(self.bucket_name)  # Ensure target bucket exists
        try:
            self.client.upload_file(str(local_file_path), self.bucket_name, key)
            s3_identifier = f"s3://{self.bucket_name}/{key}"
            logger.info(f"Successfully uploaded local file {local_file_path} to {s3_identifier}")
            return s3_identifier
        except FileNotFoundError:
            logger.error(f"Local file not found for S3 upload: {local_file_path}")
        except ClientError as e:
            logger.error(f"S3 ClientError during upload of {local_file_path} to {self.bucket_name}/{key}: {e}")
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path} to S3 {self.bucket_name}/{key}: {e}")
        return None
