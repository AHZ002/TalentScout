"""Tests for the Groq LLM service retry behaviour."""

from unittest.mock import AsyncMock

import httpx
import pytest
from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from talentscout.llm.groq import (
    GroqConnectionError,
    GroqLLMService,
    GroqRateLimitError,
    GroqRequestTooLargeError,
    GroqTimeoutError,
    GroqUpstreamError,
)


def _build_service() -> GroqLLMService:
    """Build a service instance without creating a real API client."""
    service = GroqLLMService.__new__(GroqLLMService)
    service.max_retries = 2
    service.retry_base_delay_seconds = 1.0
    service.retry_max_delay_seconds = 8.0
    return service


@pytest.mark.asyncio
async def test_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a transient timeout is retried and can subsequently succeed."""
    service = _build_service()

    operation = AsyncMock(
        side_effect=[
            APITimeoutError(request=httpx.Request("POST", "https://api.groq.com")),
            "success",
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("talentscout.llm.groq.asyncio.sleep", sleep)

    result = await service._with_retries(operation)

    assert result == "success"
    assert operation.await_count == 2
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_retry_stops_after_configured_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that retries stop after the configured maximum."""
    service = _build_service()

    error = APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))
    operation = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    monkeypatch.setattr("talentscout.llm.groq.asyncio.sleep", sleep)

    with pytest.raises(GroqTimeoutError) as exc_info:
        await service._with_retries(operation)

    assert isinstance(exc_info.value.__cause__, APITimeoutError)
    assert operation.await_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_rate_limit_retry_respects_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that the server-provided Retry-After value is respected."""
    service = _build_service()

    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "3"},
    )
    error = RateLimitError(
        "Rate limit exceeded",
        response=response,
        body=None,
    )

    operation = AsyncMock(side_effect=[error, "success"])
    sleep = AsyncMock()
    monkeypatch.setattr("talentscout.llm.groq.asyncio.sleep", sleep)

    result = await service._with_retries(operation)

    assert result == "success"
    sleep.assert_awaited_once_with(3.0)


def test_rate_limit_error_is_converted_to_safe_service_error() -> None:
    """Verify that rate-limit details are not exposed to API consumers."""
    error = RateLimitError(
        "Internal provider rate limit details",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.groq.com"),
        ),
        body=None,
    )

    result = GroqLLMService._build_service_error(error)

    assert isinstance(result, GroqRateLimitError)
    assert result.status_code == 503
    assert result.error_code == "llm_rate_limited"
    assert "Internal provider" not in result.public_message


def test_timeout_error_is_converted_to_service_error() -> None:
    """Verify that timeout errors receive the correct API classification."""
    error = APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))

    result = GroqLLMService._build_service_error(error)

    assert isinstance(result, GroqTimeoutError)
    assert result.status_code == 504


def test_connection_error_is_converted_to_service_error() -> None:
    """Verify that connection errors receive the correct API classification."""
    error = APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))

    result = GroqLLMService._build_service_error(error)

    assert isinstance(result, GroqConnectionError)
    assert result.status_code == 503


def test_upstream_error_is_converted_to_service_error() -> None:
    """Verify that Groq server errors receive the correct API classification."""
    error = InternalServerError(
        "Internal provider error",
        response=httpx.Response(
            500,
            request=httpx.Request("POST", "https://api.groq.com"),
        ),
        body=None,
    )

    result = GroqLLMService._build_service_error(error)

    assert isinstance(result, GroqUpstreamError)
    assert result.status_code == 502


@pytest.mark.asyncio
async def test_request_exceeding_token_limit_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request larger than the entire TPM limit must not be retried."""
    service = _build_service()

    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(429, request=request)

    error = RateLimitError(
        "Rate limit exceeded",
        response=response,
        body={
            "error": {
                "message": (
                    "Rate limit reached on tokens per minute (TPM): "
                    "Limit 8000, Used 0, Requested 8500."
                ),
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
    )

    operation = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    monkeypatch.setattr("talentscout.llm.groq.asyncio.sleep", sleep)

    with pytest.raises(GroqRequestTooLargeError) as exc_info:
        await service._with_retries(operation)

    assert exc_info.value.retryable is False
    assert exc_info.value.error_code == "llm_request_exceeds_token_limit"
    assert isinstance(exc_info.value.__cause__, RateLimitError)
    assert operation.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_token_window_exhaustion_is_still_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request below the TPM limit can succeed after the window resets."""
    service = _build_service()

    request = httpx.Request("POST", "https://api.groq.com")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "3"},
    )

    error = RateLimitError(
        "Rate limit exceeded",
        response=response,
        body={
            "error": {
                "message": (
                    "Rate limit reached on tokens per minute (TPM): "
                    "Limit 8000, Used 1188, Requested 7437."
                ),
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
    )

    operation = AsyncMock(side_effect=[error, "success"])
    sleep = AsyncMock()
    monkeypatch.setattr("talentscout.llm.groq.asyncio.sleep", sleep)

    result = await service._with_retries(operation)

    assert result == "success"
    assert operation.await_count == 2
    sleep.assert_awaited_once_with(3.0)
