"""Groq-backed implementation of the TalentScout LLM service."""

from typing import TypeVar, cast

from groq import AsyncGroq
from groq.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchema
from pydantic import BaseModel, ValidationError

from talentscout.config.settings import get_settings
from talentscout.llm.service import LLMService, StructuredOutputT


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when an LLM response cannot be validated against its schema."""


class GroqLLMService(LLMService):
    """Generate responses using a Groq-hosted language model."""

    def __init__(self) -> None:
        # Load the API key and model from application configuration.
        settings = get_settings()

        if not settings.groq_api_key:
            raise ValueError("TALENTSCOUT_GROQ_API_KEY is not configured")

        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response using the configured Groq model."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("The LLM returned an empty response")

        return content.strip()

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Generate a JSON-schema-constrained response and validate it locally."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format=self._build_response_format(response_model),
            temperature=0.0,
        )

        content = response.choices[0].message.content

        if not content:
            raise StructuredOutputError("The LLM returned an empty structured response")

        try:
            return response_model.model_validate_json(content)
        except ValidationError as error:
            raise StructuredOutputError(
                f"The LLM response did not match {response_model.__name__}"
            ) from error

    @staticmethod
    def _build_response_format(
        response_model: type[ResponseModelT],
    ) -> ResponseFormatResponseFormatJsonSchema:
        """Build Groq's strict JSON Schema response format for a Pydantic model."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": cast(dict[str, object], response_model.model_json_schema()),
                "strict": True,
            },
        }
