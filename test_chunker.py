"""
Unit tests for TextChunker module.
"""

import pytest
from src.chunker import TextChunker
from src.document_loader import DocumentPage, DocumentMetadata


def test_chunker_basic_splitting():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "Paragraph 1: Sentence one is short. Sentence two is longer and provides context. "
        "Paragraph 2: Sentence three is here to add characters for testing the chunker overlap."
    )
    doc_meta = DocumentMetadata(
        doc_id="doc123",
        filename="test.txt",
        file_type="TXT",
        file_size_bytes=len(text),
        total_pages=1,
        char_count=len(text),
        page_chunks=[
            DocumentPage(
                doc_id="doc123",
                filename="test.txt",
                page_number=1,
                text=text,
                total_pages=1
            )
        ]
    )

    chunks = chunker.chunk_document(doc_meta)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.doc_id == "doc123"
        assert chunk.filename == "test.txt"
        assert chunk.page_number == 1
        assert len(chunk.text) <= 120  # Allows slight boundary buffer


def test_chunker_overlap_safety():
    # Verify that overlap >= chunk_size raises ValueError
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=120)


def test_chunker_no_empty_chunks():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "Short sentence.\n\n\n   \n\nAnother sentence."

    chunks = chunker.chunk_text(text, doc_id="doc1", filename="test.txt", page_number=1)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.text.strip()) > 0


def test_chunker_metadata_preservation():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    pages = [
        DocumentPage(doc_id="doc_multi", filename="multi.pdf", page_number=1, text="Page 1 content " * 10, total_pages=2),
        DocumentPage(doc_id="doc_multi", filename="multi.pdf", page_number=2, text="Page 2 content " * 10, total_pages=2)
    ]
    doc_meta = DocumentMetadata(
        doc_id="doc_multi",
        filename="multi.pdf",
        file_type="PDF",
        file_size_bytes=500,
        total_pages=2,
        char_count=500,
        page_chunks=pages
    )

    chunks = chunker.chunk_document(doc_meta)

    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert all(c.filename == "multi.pdf" for c in chunks)
