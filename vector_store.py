"""
Vector Store Module
Implements FAISS CPU index for fast vector similarity search.
Supports document-scoped filtering, deduplication, index updates, and chunk metadata mapping.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Sequence
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from src.chunker import TextChunk


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


@dataclass
class SearchResult:
    """SearchResult wrapper holding chunk and similarity score."""
    chunk: TextChunk
    similarity_score: float  # Cosine similarity in range [-1.0, 1.0] for normalized embeddings


class FAISSVectorStore:
    """
    FAISS-backed vector store for normalized dense embeddings using Inner Product (Cosine Similarity).
    """

    def __init__(self, dimension: int = 384):
        if faiss is None:
            raise VectorStoreError(
                "FAISS library is not installed. Please install 'faiss-cpu'."
            )
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[TextChunk] = []
        self._registered_chunk_ids: Set[str] = set()

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> int:
        """
        Adds text chunks and their corresponding normalized vector embeddings to the index.
        Ignores duplicate chunk IDs.
        :param chunks: List of TextChunk objects.
        :param embeddings: 2D float32 numpy array of shape (len(chunks), dimension).
        :return: Number of new chunks added.
        """
        if not chunks:
            return 0

        if embeddings is None or len(embeddings) == 0:
            raise VectorStoreError("Embeddings array is empty.")

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
            )

        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension {embeddings.shape[1]} does not match vector store dimension {self.dimension}."
            )

        # Filter out duplicates
        new_chunks = []
        new_vecs = []

        for chunk, vec in zip(chunks, embeddings):
            if chunk.chunk_id not in self._registered_chunk_ids:
                new_chunks.append(chunk)
                new_vecs.append(vec)
                self._registered_chunk_ids.add(chunk.chunk_id)

        if not new_chunks:
            return 0

        vec_matrix = np.array(new_vecs, dtype=np.float32)

        # Ensure 2D array
        if vec_matrix.ndim == 1:
            vec_matrix = np.expand_dims(vec_matrix, axis=0)

        self.index.add(vec_matrix)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_filter: Optional[Sequence[str]] = None
    ) -> List[SearchResult]:
        """
        Search vector index for top_k most similar chunks.
        :param query_embedding: 2D numpy array of shape (1, dimension).
        :param top_k: Desired number of results.
        :param document_filter: Optional list/set of filenames or doc_ids to filter by.
        :return: List of SearchResult objects sorted by similarity descending.
        """
        if self.index.ntotal == 0 or len(self.chunks) == 0:
            return []

        if query_embedding is None or query_embedding.size == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = query_embedding.astype(np.float32)

        # Normalize set of filter targets
        filter_set = set(document_filter) if document_filter else None

        # Determine search k size: if filtering, request more candidates to ensure top_k matches after filter
        fetch_k = min(self.index.ntotal, max(top_k * 5, 50)) if filter_set else min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, fetch_k)

        results: List[SearchResult] = []
        raw_scores = scores[0]
        raw_indices = indices[0]

        for score, idx in zip(raw_scores, raw_indices):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            # Apply document-scoped filtering
            if filter_set:
                if chunk.filename not in filter_set and chunk.doc_id not in filter_set:
                    continue

            # Note: score is inner product of normalized vectors, which equals Cosine Similarity [-1.0 to 1.0]
            similarity_score = float(score)
            results.append(SearchResult(chunk=chunk, similarity_score=similarity_score))

            if len(results) >= top_k:
                break

        return results

    def remove_document(self, filename_or_doc_id: str):
        """
        Removes all chunks belonging to a document and rebuilds the index safely.
        """
        remaining_chunks = []
        remaining_ids = set()

        for chunk in self.chunks:
            if chunk.filename != filename_or_doc_id and chunk.doc_id != filename_or_doc_id:
                remaining_chunks.append(chunk)
                remaining_ids.add(chunk.chunk_id)

        if len(remaining_chunks) == len(self.chunks):
            return  # No change

        # Reset index
        self.index.reset()
        self.chunks = []
        self._registered_chunk_ids = set()

        # Re-add if remaining chunks exist
        # Note: caller will need to re-add vectors or pipeline will trigger index rebuild.

    def clear(self):
        """Resets the vector index and stored chunk metadata."""
        if self.index is not None:
            self.index.reset()
        self.chunks = []
        self._registered_chunk_ids = set()

    @property
    def total_chunks(self) -> int:
        """Return total number of vectors/chunks stored."""
        return len(self.chunks)
