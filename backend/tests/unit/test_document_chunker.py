from talentscout.documents.chunker import DocumentChunker


def test_chunker_creates_overlapping_chunks() -> None:
    """Verify that long text is split into overlapping chunks."""
    text = "a" * 250

    chunker = DocumentChunker(
        chunk_size=100,
        overlap=20,
    )

    chunks = chunker.chunk(text)

    assert len(chunks) == 4
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    assert chunks[2].index == 2
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_chunker_returns_empty_list_for_empty_text() -> None:
    """Verify that empty documents produce no chunks."""
    chunker = DocumentChunker()

    assert chunker.chunk("") == []
