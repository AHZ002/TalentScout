"""change document embedding dimension

Revision ID: 9e08e12d2be9
Revises: a963e816ac36
Create Date: 2026-09-01 21:51:38.403159

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9e08e12d2be9'
down_revision: str | Sequence[str] | None = 'a963e816ac36'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Change document chunk embeddings from 1536 to 1024 dimensions."""
    op.execute(
        """
        ALTER TABLE document_chunks
        ALTER COLUMN embedding TYPE vector(1024)
        """
    )


def downgrade() -> None:
    """Restore document chunk embeddings to 1536 dimensions."""
    op.execute(
        """
        ALTER TABLE document_chunks
        ALTER COLUMN embedding TYPE vector(1536)
        """
    )
