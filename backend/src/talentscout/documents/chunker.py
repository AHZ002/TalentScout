from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A piece of extracted document text used for retrieval."""

    text: str
    index: int


class DocumentChunker:
    """Splits extracted document text into manageable chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between zero and chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[DocumentChunk]:
        """Split text into overlapping chunks."""
        text = text.strip()

        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        index = 0
        step = self.chunk_size - self.overlap

        while start < len(text):
            chunk_text = text[start : start + self.chunk_size].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        index=index,
                    )
                )
                index += 1

            start += step

        return chunks
