from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.inference import inference_service

router = APIRouter(tags=["inference"])


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text payload to process")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata for routing")


class TextResponse(BaseModel):
    text: str
    response: str
    model_version: str


@router.post("/inference", response_model=TextResponse, status_code=status.HTTP_200_OK)
async def run_inference(payload: TextRequest) -> TextResponse:
    """Delegate text payload processing to the inference service."""

    try:
        generated_text = inference_service.generate_response(payload.text, payload.metadata)
    except Exception as exc:  # pragma: no cover - defensive guard for future integrations
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference processing failed") from exc

    return TextResponse(text=payload.text, response=generated_text, model_version=inference_service.model_version)
