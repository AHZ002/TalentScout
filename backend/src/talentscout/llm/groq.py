"""Groq-backed implementation of the TalentScout LLM service."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

from groq import (
    APIConnectionError,
    APITimeoutError,
    AsyncGroq,
    InternalServerError,
    RateLimitError,
)
from groq.types.chat.completion_create_params import (
    ResponseFormatResponseFormatJsonSchema,
)
from pydantic import BaseModel, ValidationError

from talentscout.config.settings import get_settings
from talentscout.llm.service import LLMService, StructuredOutputT

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)
ResponseT = TypeVar("ResponseT")

type RetryableGroqError = (
    RateLimitError | APITimeoutError | APIConnectionError | InternalServerError
)

RETRYABLE_GROQ_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


class StructuredOutputError(RuntimeError):
    """Raised when an LLM response cannot be validated against its schema."""


class GroqServiceError(RuntimeError):
    """Base exception for failures communicating with the Groq service."""

    status_code = 503
    error_code = "llm_service_unavailable"
    public_message = "The AI service is temporarily unavailable."
    retryable = True


class GroqRateLimitError(GroqServiceError):
    """Raised when Groq rate limits remain exhausted after retries."""

    error_code = "llm_rate_limited"
    public_message = "The AI service is temporarily rate-limited. Please try again shortly."


class GroqRequestTooLargeError(GroqServiceError):
    """Raised when one request exceeds the provider's token rate limit."""

    error_code = "llm_request_exceeds_token_limit"
    public_message = (
        "This AI request exceeds the provider's token limit. "
        "The interview context or requested response is too large."
    )
    retryable = False


class GroqTimeoutError(GroqServiceError):
    """Raised when Groq requests repeatedly time out."""

    status_code = 504
    error_code = "llm_timeout"
    public_message = "The AI service took too long to respond. Please try again."


class GroqConnectionError(GroqServiceError):
    """Raised when the application cannot connect to Groq."""

    error_code = "llm_connection_error"
    public_message = "The AI service could not be reached. Please try again."


class GroqUpstreamError(GroqServiceError):
    """Raised when Groq repeatedly returns server errors."""

    status_code = 502
    error_code = "llm_upstream_error"
    public_message = "The AI service encountered an internal error. Please try again."


class GroqLLMService(LLMService):
    """Generate responses using a Groq-hosted language model."""

    def __init__(self) -> None:
        """Initialize the Groq client using application configuration."""
        settings = get_settings()

        if not settings.groq_api_key:
            raise ValueError("TALENTSCOUT_GROQ_API_KEY is not configured")

        self.client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self.model = settings.llm_model
        self.structured_max_completion_tokens = settings.llm_structured_max_completion_tokens
        self.max_retries = settings.llm_max_retries
        self.retry_base_delay_seconds = settings.llm_retry_base_delay_seconds
        self.retry_max_delay_seconds = settings.llm_retry_max_delay_seconds

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response using the configured Groq model."""
        response = await self._with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
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
        """Generate and validate a JSON-schema-constrained response."""
        response = await self._with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self._build_response_format(response_model),
                temperature=0.0,
                max_completion_tokens=self.structured_max_completion_tokens,
            )
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
    def _build_service_error(error: RetryableGroqError) -> GroqServiceError:
        """Convert an exhausted Groq SDK error into a safe application error."""
        if isinstance(error, RateLimitError):
            return GroqRateLimitError()

        if isinstance(error, APITimeoutError):
            return GroqTimeoutError()

        if isinstance(error, APIConnectionError):
            return GroqConnectionError()

        return GroqUpstreamError()

    async def _with_retries(
        self,
        operation: Callable[[], Awaitable[ResponseT]],
    ) -> ResponseT:
        """Retry transient Groq failures using bounded exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except RETRYABLE_GROQ_ERRORS as error:
                if isinstance(error, RateLimitError) and self._request_exceeds_token_limit(error):
                    raise GroqRequestTooLargeError() from error

                if attempt >= self.max_retries:
                    raise self._build_service_error(error) from error

                delay = self._get_retry_delay(error, attempt)
                await asyncio.sleep(delay)

        raise RuntimeError("Groq retry loop exited unexpectedly")

    def _get_retry_delay(
        self,
        error: RetryableGroqError,
        attempt: int,
    ) -> float:
        """Calculate a bounded delay, respecting a server-provided retry hint."""
        retry_after = None

        response = getattr(error, "response", None)
        if response is not None:
            headers = response.headers
            retry_after_value = headers.get("retry-after-ms") or headers.get("retry-after")

            if retry_after_value is not None:
                try:
                    retry_after = float(retry_after_value)
                    if headers.get("retry-after-ms") == retry_after_value:
                        retry_after /= 1000
                except (TypeError, ValueError):
                    retry_after = None

        if retry_after is None:
            retry_after = self.retry_base_delay_seconds * (2**attempt)

        return min(retry_after, self.retry_max_delay_seconds)

    @staticmethod
    def _request_exceeds_token_limit(error: RateLimitError) -> bool:
        """Return True only when the request itself exceeds the token limit."""
        message = ""

        body = error.body
        if isinstance(body, dict):
            error_details = body.get("error")
            if isinstance(error_details, dict):
                candidate_message = error_details.get("message")
                if isinstance(candidate_message, str):
                    message = candidate_message

        if not message:
            message = str(error)

        match = re.search(
            r"\bLimit\s+(\d+)\s*,\s*Used\s+(\d+)\s*,\s*Requested\s+(\d+)\b",
            message,
            flags=re.IGNORECASE,
        )

        if match is None:
            return False

        limit = int(match.group(1))
        requested = int(match.group(3))

        return requested > limit

    @staticmethod
    def _build_response_format(
        response_model: type[ResponseModelT],
    ) -> ResponseFormatResponseFormatJsonSchema:
        """Build Groq's strict JSON Schema response format."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": cast(dict[str, object], response_model.model_json_schema()),
                "strict": True,
            },
        }
