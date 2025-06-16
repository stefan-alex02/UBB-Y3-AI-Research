import io
import logging
from pathlib import PurePath

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse

from ..core.config import APP_LOGGER_NAME
from ..persistence import ArtifactRepository

logger = logging.getLogger(APP_LOGGER_NAME)

router = APIRouter()


@router.post("/upload")
async def upload_image_endpoint(
    fast_api_request: FastAPIRequest,
    username: str = Form(None),
    image_id: str = Form(None),
    image_format: str = Form(None),
    file: UploadFile = File(None)
):
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
    else:
        logger.info("PYTHON PROCESSED file: None or not recognized")

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
    logger.debug(f"Attempting to retrieve image from Python storage: {image_key}")
    try:
        file_bytes = artifact_repo.download_file_to_memory(image_key)
        if file_bytes is None:
            logger.warning(f"Image not found in Python storage: {image_key}")
            raise HTTPException(status_code=404, detail="Image not found.")

        media_type = "application/octet-stream"
        if image_filename_with_ext.lower().endswith(".png"):
            media_type = "image/png"
        elif image_filename_with_ext.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        elif image_filename_with_ext.lower().endswith(".gif"):
            media_type = "image/gif"

        return StreamingResponse(io.BytesIO(file_bytes), media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image {image_key} from Python storage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve image from storage.")


@router.delete("/{username}/{image_id_with_format}")
async def delete_image_api(
        username: str,
        image_id_with_format: str,
        fast_api_request: FastAPIRequest
):
    logger.info(f"Received request to delete image: {image_id_with_format} for user {username}")
    artifact_repo: ArtifactRepository = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    image_key = str((PurePath("images") / username / image_id_with_format).as_posix())

    success = artifact_repo.delete_object(image_key)
    if success:
        image_id_without_ext = PurePath(image_id_with_format).stem
        predictions_prefix_for_image = str((PurePath("predictions") / username / image_id_without_ext).as_posix()) + "/"
        logger.info(
            f"Attempting to delete associated prediction artifacts under prefix: {predictions_prefix_for_image}")
        preds_deleted_success = artifact_repo.delete_objects_by_prefix(predictions_prefix_for_image)
        if not preds_deleted_success:
            logger.warning(
                f"Failed to delete all prediction artifacts for image {image_id_with_format}, but image file itself might be deleted.")

        return {"message": f"Image {image_key} and associated predictions deletion process initiated."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete image file {image_key} from artifact store.")
