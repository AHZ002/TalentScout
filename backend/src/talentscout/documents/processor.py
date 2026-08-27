from abc import ABC, abstractmethod


class DocumentProcessor(ABC):
    """Defines how TalentScout extracts text from documents."""

    @abstractmethod
    async def extract_text(
        self,
        content: bytes,
        content_type: str,
    ) -> str:
        """Extract readable text from a document."""
        raise NotImplementedError
