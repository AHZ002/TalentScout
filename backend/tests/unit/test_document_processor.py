from io import BytesIO

import pytest
from pypdf import PdfWriter

from talentscout.documents.processors.basic import BasicDocumentProcessor


@pytest.mark.asyncio
async def test_extract_text_from_plain_text() -> None:
    """Verify that plain text documents are extracted correctly."""
    processor = BasicDocumentProcessor()

    result = await processor.extract_text(
        b"Patient risk prediction guidelines.",
        "text/plain",
    )

    assert result == "Patient risk prediction guidelines."


@pytest.mark.asyncio
async def test_reject_unsupported_document_type() -> None:
    """Verify that unsupported document types are rejected."""
    processor = BasicDocumentProcessor()

    with pytest.raises(ValueError, match="Unsupported document type"):
        await processor.extract_text(
            b"unsupported",
            "application/octet-stream",
        )


@pytest.mark.asyncio
async def test_extract_text_from_pdf() -> None:
    """Verify that text can be extracted from a PDF document."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)

    processor = BasicDocumentProcessor()

    result = await processor.extract_text(
        pdf_buffer.getvalue(),
        "application/pdf",
    )

    assert isinstance(result, str)
