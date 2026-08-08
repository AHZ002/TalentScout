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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
