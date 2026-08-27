from sqlalchemy.orm import (
    DeclarativeBase,
)  # SQLAlchemy class that gives you a foundation for defining database models using Python classes.


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy database models."""
