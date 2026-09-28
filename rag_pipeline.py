"""
RAG Pipeline Module
Orchestrates Document Loader, Text Chunker, Embedding Service, FAISS Vector Store,
and LLM Service into a unified Retrieval-Augmented Generation workflow.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict, Any, Union, BinaryIO
from src.document_loader import DocumentLoader, DocumentMetadata, DocumentProcessingError
from src.chunker import TextChunker, TextChunk
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore, SearchResult
from src.llm_service import LLMService, LLMError, MissingAPIKeyError


@dataclass
class RAGResponse:
    """Structure representing the result of a RAG query execution."""
    question: str
    answer: str
    source_results: List[SearchResult] = field(default_factory=list)
    status: str = "success"  # "success", "insufficient_context", "no_documents", "error"
    llm_called: bool = False
    error_message: Optional[str] = None


class RAGPipeline:
    """Complete RAG pipeline managing ingestion, embedding, vector retrieval, and LLM answer generation."""

    INSUFFICIENT_CONTEXT_MSG = "I could not find sufficient information in the uploaded documents to answer this question."

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        min_similarity_threshold: float = 0.15
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_similarity_threshold = min_similarity_threshold

        self.document_loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_service = EmbeddingService(model_name=embedding_model_name)
        self.vector_store = FAISSVectorStore(dimension=self.embedding_service.dimension)
        self.llm_service = LLMService(api_key=api_key, model_name=llm_model_name)

        # Registry of ingested documents: filename -> DocumentMetadata
        self.ingested_documents: Dict[str, DocumentMetadata] = {}

    def set_api_key(self, api_key: str):
        """Update the Gemini API Key dynamically."""
        self.llm_service.api_key = api_key

    def update_chunking_config(self, chunk_size: int, chunk_overlap: int):
        """Update chunker parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_document(
        self,
        filename: str,
        file_source: Union[bytes, BinaryIO, str]
    ) -> DocumentMetadata:
        """
        Ingest a single document: extract text, split into chunks, compute embeddings, and index in FAISS.
        """
        # If document already ingested under same filename, remove old vectors first
        if filename in self.ingested_documents:
            self.remove_document(filename)

        doc_metadata = self.document_loader.load_document(filename, file_source)
        chunks = self.chunker.chunk_document(doc_metadata)

        if not chunks:
            raise DocumentProcessingError(f"Document '{filename}' yielded 0 valid text chunks.")

        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        self.vector_store.add_chunks(chunks, embeddings)
        self.ingested_documents[filename] = doc_metadata

        return doc_metadata

    def remove_document(self, filename: str):
        """Remove a document from the ingested registry and rebuild index."""
        if filename in self.ingested_documents:
            del self.ingested_documents[filename]

        # Rebuild whole vector store from remaining ingested documents
        remaining_docs = list(self.ingested_documents.values())
        self.vector_store.clear()

        for doc_meta in remaining_docs:
            chunks = self.chunker.chunk_document(doc_meta)
            if chunks:
                chunk_texts = [c.text for c in chunks]
                embeddings = self.embedding_service.embed_texts(chunk_texts)
                self.vector_store.add_chunks(chunks, embeddings)

    def clear_all(self):
        """Reset all state, ingested documents, and vector index."""
        self.ingested_documents.clear()
        self.vector_store.clear()

    def query(
        self,
        question: str,
        selected_documents: Optional[Sequence[str]] = None,
        top_k: int = 5
    ) -> RAGResponse:
        """
        Execute RAG query against stored document index.
        :param question: User question string.
        :param selected_documents: List of document filenames to search against. If None, searches all.
        :param top_k: Maximum number of relevant passages to retrieve.
        :return: RAGResponse object.
        """
        question = question.strip() if question else ""
        if not question:
            return RAGResponse(
                question="",
                answer="Please enter a valid question.",
                status="error",
                llm_called=False,
                error_message="Question is empty."
            )

        if self.vector_store.total_chunks == 0:
            return RAGResponse(
                question=question,
                answer="No documents have been uploaded and processed yet. Please upload a PDF or TXT file first.",
                status="no_documents",
                llm_called=False
            )

        # Step 1: Embed question
        try:
            query_emb = self.embedding_service.embed_query(question)
        except Exception as e:
            return RAGResponse(
                question=question,
                answer=f"Failed to generate embedding for the question: {str(e)}",
                status="error",
                llm_called=False,
                error_message=str(e)
            )

        # Step 2: Search FAISS Index
        search_results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=top_k,
            document_filter=selected_documents
        )

        # Step 3: Check relevance threshold
        filtered_results = [
            res for res in search_results
            if res.similarity_score >= self.min_similarity_threshold
        ]

        if not filtered_results:
            return RAGResponse(
                question=question,
                answer="No sufficiently relevant supporting passage was found in the selected documents to answer this question.",
                source_results=search_results,
                status="insufficient_context",
                llm_called=False
            )

        # Step 4: Construct context block
        context_blocks = []
        for i, res in enumerate(filtered_results, 1):
            page_info = f" (Page {res.chunk.page_number})" if res.chunk.page_number else ""
            context_blocks.append(
                f"[Source {i}: {res.chunk.filename}{page_info}]\n{res.chunk.text}"
            )

        formatted_context = "\n\n".join(context_blocks)

        # Step 5: Generate answer using LLM
        try:
            answer = self.llm_service.generate_answer(
                question=question,
                context_str=formatted_context
            )
            return RAGResponse(
                question=question,
                answer=answer,
                source_results=filtered_results,
                status="success",
                llm_called=True
            )
        except MissingAPIKeyError as e:
            return RAGResponse(
                question=question,
                answer=(
                    f"**Gemini API Key Required**\n\n"
                    f"{str(e)}\n\n"
                    f"Below are the top retrieved passages from your uploaded documents:"
                ),
                source_results=filtered_results,
                status="error",
                llm_called=False,
                error_message=str(e)
            )
        except LLMError as e:
            return RAGResponse(
                question=question,
                answer=f"LLM Answer Generation Failed: {str(e)}",
                source_results=filtered_results,
                status="error",
                llm_called=False,
                error_message=str(e)
            )
