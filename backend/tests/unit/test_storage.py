from pathlib import Path

import pytest

from talentscout.storage.local import LocalStorageService


@pytest.mark.asyncio
async def test_local_storage_saves_file(tmp_path: Path) -> None:
    """Verify that uploaded content is saved to local storage."""
    storage = LocalStorageService(tmp_path)

    storage_path = await storage.save(
        content=b"Hello TalentScout",
        filename="example.txt",
        content_type="text/plain",
    )

    saved_file = Path(storage_path)

    assert saved_file.exists()
    assert saved_file.read_bytes() == b"Hello TalentScout"
    assert saved_file.suffix == ".txt"


@pytest.mark.asyncio
async def test_local_storage_uses_safe_generated_filename(
    tmp_path: Path,
) -> None:
    """Verify that the original filename cannot control the storage path."""
    storage = LocalStorageService(tmp_path)

    storage_path = await storage.save(
        content=b"safe content",
        filename="../../unsafe.txt",
        content_type="text/plain",
    )

    saved_file = Path(storage_path)

    assert saved_file.parent == tmp_path
    assert saved_file.name != "../../unsafe.txt"
    assert saved_file.read_bytes() == b"safe content"


@pytest.mark.asyncio
async def test_local_storage_deletes_file(tmp_path: Path) -> None:
    """Verify that a stored file can be deleted."""
    storage = LocalStorageService(tmp_path)

    storage_path = await storage.save(
        content=b"delete me",
        filename="example.txt",
        content_type="text/plain",
    )

    saved_file = Path(storage_path)
    assert saved_file.exists()

    await storage.delete(storage_path)

    assert not saved_file.exists()
