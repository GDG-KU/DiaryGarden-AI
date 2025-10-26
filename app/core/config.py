from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    """Application configuration that can be overridden via environment variables."""

    project_name: str = "DiaryGarden AI Ops API"
    api_version: str = "0.1.0"
    debug: bool = False

    @classmethod
    def load(cls) -> Settings:
        return cls(
            project_name=os.getenv("PROJECT_NAME", cls.project_name),
            api_version=os.getenv("API_VERSION", cls.api_version),
            debug=os.getenv("DEBUG", str(cls.debug)).lower() in {"1", "true", "yes"},
        )


settings = Settings.load()
