"""
Text Chunker Module
Implements semantic character/paragraph chunking with configurable overlap.
Preserves rich metadata (filename, page number, chunk index, document ID) for every chunk.
Guarantees loop safety and prevents zero-length chunks.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional
from src.document_loader import DocumentPage, DocumentMetadata


@dataclass
class TextChunk:
    """Represents a text chunk with associated document metadata."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    char_count: int


class TextChunker:
    """
    Splits document text into manageable, overlapping chunks while respecting
    paragraph and sentence boundaries where practical.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size to prevent infinite loops.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_metadata: DocumentMetadata) -> List[TextChunk]:
        """
        Processes all pages in a document and returns a flat list of TextChunk objects.
        """
        all_chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in doc_metadata.page_chunks:
            page_chunks = self.chunk_text(
                text=page.text,
                doc_id=page.doc_id,
                filename=page.filename,
                page_number=page.page_number,
                start_chunk_index=global_chunk_idx
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        page_number: Optional[int] = None,
        start_chunk_index: int = 0
    ) -> List[TextChunk]:
        """
        Chunks a given block of text using boundary-aware recursive splitting.
        """
        text = text.strip()
        if not text:
            return []

        # If text is smaller than chunk size, return single chunk immediately
        if len(text) <= self.chunk_size:
            chunk_id = self._make_chunk_id(doc_id, page_number, start_chunk_index, text)
            return [
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_index=start_chunk_index,
                    text=text,
                    char_count=len(text)
                )
            ]

        chunks: List[TextChunk] = []
        step_size = self.chunk_size - self.chunk_overlap
        pos = 0
        text_length = len(text)
        chunk_counter = start_chunk_index

        # Natural boundary separators in order of preference
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]

        while pos < text_length:
            target_end = pos + self.chunk_size

            if target_end >= text_length:
                chunk_text = text[pos:].strip()
                if chunk_text:
                    chunk_id = self._make_chunk_id(doc_id, page_number, chunk_counter, chunk_text)
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page_number,
                            chunk_index=chunk_counter,
                            text=chunk_text,
                            char_count=len(chunk_text)
                        )
                    )
                break

            # Find best boundary before target_end
            best_split = -1
            sub_window = text[pos:target_end]

            # Look for boundary separators from the end of the window backwards
            for sep in separators:
                last_sep_pos = sub_window.rfind(sep)
                # Ensure the split position leaves at least a reasonable chunk length (e.g. > 1/3 chunk_size)
                if last_sep_pos > int(self.chunk_size * 0.3):
                    best_split = pos + last_sep_pos + len(sep)
                    break

            # Fallback if no natural boundary found
            if best_split <= pos:
                best_split = target_end

            chunk_text = text[pos:best_split].strip()
            if chunk_text:
                chunk_id = self._make_chunk_id(doc_id, page_number, chunk_counter, chunk_text)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page_number,
                        chunk_index=chunk_counter,
                        text=chunk_text,
                        char_count=len(chunk_text)
                    )
                )
                chunk_counter += 1

            # Advance pos ensuring loop forward progress
            next_pos = max(best_split - self.chunk_overlap, pos + 1)
            # Guard against staying at same pos or moving backwards
            if next_pos <= pos:
                next_pos = pos + max(1, step_size)
            pos = next_pos

        return chunks

    @staticmethod
    def _make_chunk_id(doc_id: str, page_number: Optional[int], index: int, text: str) -> str:
        page_str = f"p{page_number}" if page_number is not None else "p0"
        hash_suffix = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
        return f"{doc_id}_{page_str}_c{index}_{hash_suffix}"
