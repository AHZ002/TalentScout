import pytest
from sqlalchemy import text

from talentscout.db.session import SessionFactory


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_connection() -> None:
    """Verify that the application can connect to PostgreSQL."""
    async with SessionFactory() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1
