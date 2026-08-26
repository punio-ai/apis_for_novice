from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    ollama_base_url: str
    embedding_model: str
    embedding_dimensions: int

    class Config:
        env_file = ".env"


settings = Settings()
