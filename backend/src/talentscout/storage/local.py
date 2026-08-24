from pathlib import Path
from uuid import uuid4

from talentscout.storage.base import StorageService


class LocalStorageService(StorageService):
    """Stores uploaded files on the local filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def save(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        """Save a file locally and return its relative storage path."""
        del content_type

        self.root.mkdir(parents=True, exist_ok=True)

        # Keep the original extension but generate our own safe filename.
        suffix = Path(filename).suffix.lower()
        stored_filename = f"{uuid4()}{suffix}"

        file_path = self.root / stored_filename
        file_path.write_bytes(content)

        return str(file_path)

    async def delete(self, storage_path: str) -> None:
        """Delete a locally stored file."""
        file_path = Path(storage_path)

        if file_path.exists():
            file_path.unlink()
