"""
Document Loader Module
Handles document ingestion, format validation, text extraction (PyMuPDF for PDF, UTF-8 for TXT),
and text normalization while preserving document metadata.
"""

import os
import re
import hashlib
import io
from dataclasses import dataclass, field
from typing import List, Optional, Union, BinaryIO

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when an unsupported file format is provided."""
    pass


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a document contains no extractable text."""
    pass


class FileTooLargeError(DocumentProcessingError):
    """Raised when a file exceeds the maximum allowed size."""
    pass


@dataclass
class DocumentPage:
    """Represents a single page or section of an extracted document."""
    doc_id: str
    filename: str
    page_number: Optional[int]  # 1-indexed for PDFs, 1 or None for TXT
    text: str
    total_pages: int


@dataclass
class DocumentMetadata:
    """Summary metadata for an ingested document."""
    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    total_pages: int
    char_count: int
    page_chunks: List[DocumentPage] = field(default_factory=list)


class DocumentLoader:
    """Document loader supporting PDF and TXT file ingestion with validation and normalization."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
    DEFAULT_MAX_FILE_SIZE_MB = 25

    def __init__(self, max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB):
        self.max_file_size_mb = max_file_size_mb
        self.max_bytes = int(max_file_size_mb * 1024 * 1024)

    @staticmethod
    def generate_doc_id(filename: str, content: bytes) -> str:
        """Generate a deterministic unique ID based on filename and content hash."""
        hasher = hashlib.sha256()
        hasher.update(filename.encode("utf-8", errors="replace"))
        hasher.update(content)
        return hasher.hexdigest()[:16]

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text whitespace without destroying paragraph boundaries.
        Converts Windows CRLF to LF, merges trailing spaces, and caps consecutive newlines at 2.
        """
        if not text:
            return ""
        # Replace CRLF with LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace tab or consecutive spaces with a single space (leaving newlines)
        text = re.sub(r"[ \t]+", " ", text)
        # Clean lines: strip whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        # Rejoin lines, capping consecutive empty lines to a max of 2 (\n\n)
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def validate_file(self, filename: str, file_bytes: bytes) -> str:
        """Validate extension and file size."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file format '{ext}'. Only .pdf and .txt files are supported."
            )

        if len(file_bytes) > self.max_bytes:
            raise FileTooLargeError(
                f"File '{filename}' exceeds maximum allowed size of {self.max_file_size_mb} MB "
                f"({len(file_bytes) / (1024 * 1024):.2f} MB)."
            )

        if len(file_bytes) == 0:
            raise EmptyDocumentError(f"File '{filename}' is completely empty (0 bytes).")

        return ext

    def load_document(self, filename: str, file_source: Union[bytes, BinaryIO, str]) -> DocumentMetadata:
        """
        Ingest and extract text from a PDF or TXT document.
        :param filename: Original name of the document.
        :param file_source: File path, bytes, or file-like object.
        :return: DocumentMetadata containing pages and metadata.
        """
        file_bytes = self._read_bytes(file_source)
        ext = self.validate_file(filename, file_bytes)
        doc_id = self.generate_doc_id(filename, file_bytes)

        if ext == ".pdf":
            pages = self._load_pdf(filename, doc_id, file_bytes)
        elif ext == ".txt":
            pages = self._load_txt(filename, doc_id, file_bytes)
        else:
            raise UnsupportedFormatError(f"Unsupported format: {ext}")

        total_char_count = sum(len(p.text) for p in pages)
        if total_char_count == 0:
            raise EmptyDocumentError(
                f"No extractable text found in '{filename}'. "
                f"If this is a scanned PDF, optical character recognition (OCR) is required."
            )

        return DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=ext[1:].upper(),
            file_size_bytes=len(file_bytes),
            total_pages=len(pages),
            char_count=total_char_count,
            page_chunks=pages
        )

    def _read_bytes(self, file_source: Union[bytes, BinaryIO, str]) -> bytes:
        if isinstance(file_source, bytes):
            return file_source
        elif isinstance(file_source, str):
            if os.path.exists(file_source):
                with open(file_source, "rb") as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"File not found at path: {file_source}")
        elif hasattr(file_source, "read"):
            content = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            return content
        else:
            raise ValueError("Unsupported file source format.")

    def _load_pdf(self, filename: str, doc_id: str, file_bytes: bytes) -> List[DocumentPage]:
        if fitz is None:
            raise DocumentProcessingError(
                "PyMuPDF (fitz) is not installed. PDF processing is unavailable."
            )

        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to parse PDF document '{filename}': {str(e)}")

        total_pages = len(pdf_doc)
        if total_pages == 0:
            pdf_doc.close()
            raise EmptyDocumentError(f"PDF document '{filename}' contains 0 pages.")

        pages = []
        for page_idx in range(total_pages):
            try:
                page = pdf_doc.load_page(page_idx)
                raw_text = page.get_text("text")
                normalized = self.normalize_text(raw_text)
                if normalized:
                    pages.append(
                        DocumentPage(
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page_idx + 1,  # 1-indexed
                            text=normalized,
                            total_pages=total_pages
                        )
                    )
            except Exception as e:
                # Log error per page without crashing whole document if possible
                continue

        pdf_doc.close()
        return pages

    def _load_txt(self, filename: str, doc_id: str, file_bytes: bytes) -> List[DocumentPage]:
        # Try encodings sequentially
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        raw_text = None

        for enc in encodings:
            try:
                raw_text = file_bytes.decode(enc)
                break
            except (UnicodeDecodeError, AttributeError):
                continue

        if raw_text is None:
            # Fallback with replacement characters
            raw_text = file_bytes.decode("utf-8", errors="replace")

        normalized = self.normalize_text(raw_text)
        if not normalized:
            return []

        return [
            DocumentPage(
                doc_id=doc_id,
                filename=filename,
                page_number=1,
                text=normalized,
                total_pages=1
            )
        ]
