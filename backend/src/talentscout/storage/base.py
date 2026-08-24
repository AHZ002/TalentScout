from abc import ABC, abstractmethod


class StorageService(ABC):
    """Defines the interface for storing uploaded files."""

    @abstractmethod
    async def save(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        """Store a file and return its storage path."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, storage_path: str) -> None:
        """Delete a stored file."""
        raise NotImplementedError
