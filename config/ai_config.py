from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional


class AIConfig(BaseSettings):
    AI_PROVIDER: Literal['ANTHROPIC', 'GEMINI'] = 'ANTHROPIC'
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

ai_config = AIConfig()
    