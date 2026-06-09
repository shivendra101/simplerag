from pydantic_settings import BaseSettings, SettingsConfigDict

class EmbeddingConfig(BaseSettings):
    EMBEDDING_MODEL_API_KEY: str
    EMBEDDING_MODEL: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

embedding_config = EmbeddingConfig()