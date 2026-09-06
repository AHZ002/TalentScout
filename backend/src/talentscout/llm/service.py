"""Provider-independent interface for language model services."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class LLMService(ABC):
    """Define the interface used by TalentScout agents to call an LLM."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a text response from the language model."""
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Generate and validate a response against a Pydantic model."""
        raise NotImplementedError
