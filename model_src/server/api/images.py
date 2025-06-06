from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
import io
import logging
from pathlib import PurePath
from pydantic import BaseModel

# from app.main import artifact_repo_instance # Avoid global
import logging
from ..core.config import APP_LOGGER_NAME # Import the consistent name

logger = logging.getLogger(APP_LOGGER_NAME) # Use the same name

router = APIRouter()


@router.post("/upload")
async def upload_image_endpoint(
    fast_api_request: FastAPIRequest,
    username: str = Form(None), # Temporarily optional for debugging
    image_id: str = Form(None),
    image_format: str = Form(None),
    file: UploadFile = File(None)
):
    # Log the raw form data
    try:
        form_data = await fast_api_request.form()
        logger.info(f"PYTHON RAW FORM DATA RECEIVED: {form_data}")
        logger.info(f"PYTHON FORM KEYS: {list(form_data.keys())}")
    except Exception as e:
        logger.error(f"PYTHON Error reading form data: {e}")

    logger.info(f"PYTHON PROCESSED username: {username}")
    logger.info(f"PYTHON PROCESSED image_id: {image_id}")
    logger.info(f"PYTHON PROCESSED image_format: {image_format}")
    if file:
        logger.info(f"PYTHON PROCESSED file: filename='{file.filename}', content_type='{file.content_type}', size={file.size}")
        # # To inspect first few bytes if needed (be careful with large files)
        # content_sample = await file.read(100)
        # logger.info(f"PYTHON file content sample (first 100 bytes): {content_sample}")
        # await file.seek(0) # Reset read pointer if you read from it
    else:
        logger.info("PYTHON PROCESSED file: None or not recognized")

    # Your original validation and logic
    if not username or not image_id or not image_format or not file:
        missing = []
        if not username: missing.append("username")
        if not image_id: missing.append("image_id")
        if not image_format: missing.append("image_format")
        if not file: missing.append("file")
        logger.error(f"Python endpoint: Missing required form fields: {', '.join(missing)}. Form data was: {form_data if 'form_data' in locals() else 'Error reading form'}")
        raise HTTPException(status_code=422, detail=f"Missing required form fields: {', '.join(missing)}")

    filename = f"{image_id}.{image_format.lower()}"
    image_key = str((PurePath("images") / username / filename).as_posix())

    logger.info(f"Attempting to upload image for user {username} as {image_key}")
    try:
        artifact_repo = fast_api_request.app.state.artifact_repo
        contents = await file.read()
        # Assuming save_image_object(image_bytes, key, content_type)
        content_type = file.content_type if file.content_type else 'application/octet-stream'
        saved_path = artifact_repo.save_image_object(contents, image_key, content_type=content_type)
        if not saved_path:
            raise HTTPException(status_code=500, detail="Failed to save image to artifact store.")
        return {"message": "Image uploaded successfully", "image_key": image_key, "saved_path": saved_path}
    except Exception as e:
        logger.error(f"Error uploading image {filename} for user {username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image upload failed: {e}")


@router.get("/{username}/{image_filename_with_ext}")
async def get_image_endpoint(username: str, image_filename_with_ext: str, fast_api_request: FastAPIRequest):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    image_key = f"images/{username}/{image_filename_with_ext}"
    logger.debug(f"Attempting to retrieve image: {image_key}")
    try:
        file_bytes = artifact_repo.download_file_to_memory(image_key)
        if file_bytes is None:
            raise HTTPException(status_code=404, detail="Image not found.")

        media_type = "application/octet-stream"
        if image_filename_with_ext.lower().endswith(".png"):
            media_type = "image/png"
        elif image_filename_with_ext.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"

        return StreamingResponse(io.BytesIO(file_bytes), media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image {image_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve image.")

# TODO: DELETE endpoint for images
# Needs to delete from MinIO/local file system.
# @router.delete("/{username}/{image_id_with_format}") ...
