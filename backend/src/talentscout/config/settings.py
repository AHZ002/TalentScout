from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "testing", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TALENTSCOUT_",
        case_sensitive=False,
        extra="forbid",
    )

    app_name: str = Field(
        default="TalentScout API",
        min_length=1,
    )
    app_version: str = Field(
        default="0.1.0",
        min_length=1,
    )
    environment: Environment = "development"
    debug: bool = False
    database_url: str = Field(
        default="postgresql+psycopg://talentscout:talentscout@localhost:5432/talentscout",
        min_length=1,
    )

    # Browser origins allowed to call the API.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
    )

    # API key used to generate document embeddings with Jina AI.
    jina_api_key: str = Field(default="")

    # Embedding model used for semantic document retrieval.
    embedding_model: str = Field(
        default="jina-embeddings-v5-text-small",
    )

    # Must match the VECTOR dimension in the document_chunks table.
    embedding_dimensions: int = Field(
        default=1024,
        ge=1,
    )
    # API key used by the interview agents to access the Groq LLM.
    groq_api_key: str = Field(default="")

    # LLM used for interview question generation and later agent tasks.
    llm_model: str = Field(default="openai/gpt-oss-20b")

    # Maximum number of retries after the initial Groq request.
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # Maximum time allowed for one Groq API request, in seconds.
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)

    # Maximum output tokens for structured Groq responses.
    llm_structured_max_completion_tokens: int = Field(
        default=2048,
        ge=256,
        le=4096,
    )

    # Initial and maximum delays between retry attempts, in seconds.
    llm_retry_base_delay_seconds: float = Field(default=1.0, gt=0, le=30)
    llm_retry_max_delay_seconds: float = Field(default=8.0, gt=0, le=60)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
