"""Groq-backed implementation of the TalentScout LLM service."""

from groq import AsyncGroq

from talentscout.config.settings import get_settings
from talentscout.llm.service import LLMService


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
