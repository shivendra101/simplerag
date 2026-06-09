from pydantic_settings import BaseSettings

class DBConfig(BaseSettings):
    DB_USER: str
    DB_PASSWORD: str
    DB_DATABASE: str
    DB_HOST: str
    DB_PORT: int

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

db_config = DBConfig()