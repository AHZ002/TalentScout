"""Provider-independent interface for language model services."""

from abc import ABC, abstractmethod


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
