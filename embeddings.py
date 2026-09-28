"""
Embedding Service Module
Generates dense vector embeddings using Sentence Transformers.
Loads and caches embedding models, normalizes vectors for Cosine Similarity,
and processes texts in efficient batches.
"""

from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingService:
    """Service for loading SentenceTransformer model and embedding text chunks."""

    _model_instances = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        if SentenceTransformer is None:
            raise EmbeddingError(
                "sentence-transformers is not installed. Please install it using 'pip install sentence-transformers'."
            )

        if self.model_name not in EmbeddingService._model_instances:
            try:
                # Load model onto CPU explicitly for faiss-cpu compatibility
                EmbeddingService._model_instances[self.model_name] = SentenceTransformer(
                    self.model_name, device="cpu"
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load sentence-transformer model '{self.model_name}': {str(e)}"
                )

        self._model = EmbeddingService._model_instances[self.model_name]
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the loaded model."""
        model = self._get_model()
        # Test encode a dummy string to get exact dimension
        dummy_vec = model.encode("test", convert_to_numpy=True)
        return int(dummy_vec.shape[0])

    def embed_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Embed a list of text strings.
        :param texts: List of strings to encode.
        :param batch_size: Number of texts per batch.
        :param normalize: Normalize L2 norm so inner product equals cosine similarity.
        :return: 2D numpy array of shape (len(texts), dimension) and dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._get_model()

        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = np.expand_dims(embeddings, axis=0)
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Error generating text embeddings: {str(e)}")

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query string.
        :return: 2D numpy array of shape (1, dimension) and dtype float32.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")
        embeddings = self.embed_texts([query], batch_size=1, normalize=normalize)
        return embeddings
