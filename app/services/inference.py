from __future__ import annotations

from typing import Any, Optional


class InferenceService:
    """Placeholder inference service that will host the fine-tuned model."""

    def __init__(self, model_version: str = "stub-0.1") -> None:
        self.model_version = model_version

    def generate_response(self, text: str, metadata: Optional[dict[str, Any]] = None) -> str:
        # TODO: Replace with real model inference call once the model is integrated.
        formatted_metadata = f" | metadata={metadata}" if metadata else ""
        return f"Echo: {text}{formatted_metadata}"


inference_service = InferenceService()
