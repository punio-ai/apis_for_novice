from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    ollama_base_url: str
    embedding_model: str
    embedding_dimensions: int

    # Langfuse configuration (Optional, defaults to None if not in .env)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = "https://cloud.langfuse.com"

    # This ensures Pydantic reads the .env file and ignores extra fields safely
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
