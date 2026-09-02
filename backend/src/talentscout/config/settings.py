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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
