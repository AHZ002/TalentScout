from io import BytesIO

from pypdf import PdfReader

from talentscout.documents.processor import DocumentProcessor


class BasicDocumentProcessor(DocumentProcessor):
    """Extracts text from supported document formats."""

    async def extract_text(
        self,
        content: bytes,
        content_type: str,
    ) -> str:
        """Extract text from a supported document."""

        if content_type == "text/plain":
            return content.decode("utf-8")

        if content_type == "application/pdf":
            reader = PdfReader(BytesIO(content))

            pages = [page.extract_text() or "" for page in reader.pages]

            return "\n\n".join(pages)

        raise ValueError(f"Unsupported document type: {content_type}")
