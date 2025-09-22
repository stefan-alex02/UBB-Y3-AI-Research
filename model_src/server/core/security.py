from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from .config import settings

API_KEY_NAME = "X-Internal-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == settings.INTERNAL_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials (Invalid Internal API Key)",
        )
    