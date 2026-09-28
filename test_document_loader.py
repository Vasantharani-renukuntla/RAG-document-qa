"""
Unit tests for DocumentLoader module.
"""

import pytest
import os
import fitz  # PyMuPDF
from src.document_loader import (
    DocumentLoader,
    UnsupportedFormatError,
    EmptyDocumentError,
    FileTooLargeError
)


def create_minimal_pdf_bytes(text_content: str = "Hello PyMuPDF PDF test!") -> bytes:
    """Helper function to generate valid PDF bytes in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text_content)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_txt_loader_valid():
    loader = DocumentLoader()
    content = "Line 1: Hello World.\n\nLine 2: RAG Architecture test.\r\nLine 3: Clean end."
    txt_bytes = content.encode("utf-8")

    meta = loader.load_document("sample.txt", txt_bytes)

    assert meta.filename == "sample.txt"
    assert meta.file_type == "TXT"
    assert meta.total_pages == 1
    assert "Hello World." in meta.page_chunks[0].text
    assert "\r\n" not in meta.page_chunks[0].text  # Verify CRLF normalized


def test_pdf_loader_valid():
    loader = DocumentLoader()
    pdf_bytes = create_minimal_pdf_bytes("This is page one of test PDF.")

    meta = loader.load_document("test_doc.pdf", pdf_bytes)

    assert meta.filename == "test_doc.pdf"
    assert meta.file_type == "PDF"
    assert meta.total_pages == 1
    assert "This is page one" in meta.page_chunks[0].text
    assert meta.page_chunks[0].page_number == 1


def test_unsupported_format_error():
    loader = DocumentLoader()
    fake_docx_bytes = b"PK\x03\x04 fake docx"

    with pytest.raises(UnsupportedFormatError) as exc_info:
        loader.load_document("document.docx", fake_docx_bytes)

    assert "Only .pdf and .txt files are supported" in str(exc_info.value)


def test_empty_file_error():
    loader = DocumentLoader()
    empty_bytes = b""

    with pytest.raises(EmptyDocumentError):
        loader.load_document("empty.txt", empty_bytes)


def test_file_too_large_error():
    loader = DocumentLoader(max_file_size_mb=0.001)  # 1 KB limit
    large_bytes = b"A" * 2000

    with pytest.raises(FileTooLargeError):
        loader.load_document("big.txt", large_bytes)


def test_whitespace_normalization():
    raw_text = "  Header   Text  \n\n\n\n  Paragraph 1   text  \r\n\r\n  Paragraph 2   text  "
    normalized = DocumentLoader.normalize_text(raw_text)

    assert "\r" not in normalized
    assert "\n\n\n" not in normalized
    assert "Header Text" in normalized
    assert "Paragraph 1 text" in normalized
