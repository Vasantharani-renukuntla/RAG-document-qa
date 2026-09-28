"""
Unit tests for RAG Pipeline module using mocked LLM responses.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.rag_pipeline import RAGPipeline
from src.document_loader import DocumentProcessingError


@pytest.fixture
def mock_pipeline():
    """Create a RAGPipeline instance with mocked LLM API call."""
    pipeline = RAGPipeline(
        chunk_size=300,
        chunk_overlap=50,
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name="gemini-2.5-flash",
        api_key="mock_test_key",
        min_similarity_threshold=0.1
    )
    # Mock LLM service answer generation
    pipeline.llm_service.generate_answer = MagicMock(
        return_value="RAG stands for Retrieval-Augmented Generation according to the notes."
    )
    return pipeline


def test_ingest_and_retrieve(mock_pipeline):
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances LLMs "
        "by combining similarity search with vector databases like FAISS and Sentence Transformers. "
        "It prevents hallucinations by supplying verified reference context."
    )
    doc_meta = mock_pipeline.ingest_document("rag_notes.txt", sample_text.encode("utf-8"))

    assert doc_meta.filename == "rag_notes.txt"
    assert "rag_notes.txt" in mock_pipeline.ingested_documents
    assert mock_pipeline.vector_store.total_chunks > 0

    response = mock_pipeline.query("What is Retrieval-Augmented Generation?")

    assert response.status == "success"
    assert response.llm_called is True
    assert len(response.source_results) > 0
    assert response.source_results[0].chunk.filename == "rag_notes.txt"
    assert "RAG stands for Retrieval-Augmented Generation" in response.answer


def test_empty_question_validation(mock_pipeline):
    response = mock_pipeline.query("   ")
    assert response.status == "error"
    assert response.llm_called is False
    assert "valid question" in response.answer


def test_query_no_documents_processed(mock_pipeline):
    # Pipeline without any ingested documents
    response = mock_pipeline.query("What is python?")
    assert response.status == "no_documents"
    assert response.llm_called is False
    assert "No documents have been uploaded" in response.answer


def test_insufficient_context_handling(mock_pipeline):
    sample_text = "Cooking recipe for making chocolate chip cookies with flour, sugar, and butter."
    mock_pipeline.ingest_document("cookies.txt", sample_text.encode("utf-8"))

    # Set threshold high to simulate no matching context
    mock_pipeline.min_similarity_threshold = 0.95

    response = mock_pipeline.query("What is quantum computing algorithm complexity?")

    assert response.status == "insufficient_context"
    assert response.llm_called is False
    assert "No sufficiently relevant supporting passage" in response.answer


def test_document_scoped_filtering(mock_pipeline):
    doc1_text = "Python is a high-level programming language created by Guido van Rossum."
    doc2_text = "Java is a class-based object-oriented programming language developed by Sun Microsystems."

    mock_pipeline.ingest_document("python_info.txt", doc1_text.encode("utf-8"))
    mock_pipeline.ingest_document("java_info.txt", doc2_text.encode("utf-8"))

    # Query only document python_info.txt
    response = mock_pipeline.query(
        "Who created Python?",
        selected_documents=["python_info.txt"]
    )

    assert response.status == "success"
    assert len(response.source_results) > 0
    for res in response.source_results:
        assert res.chunk.filename == "python_info.txt"
