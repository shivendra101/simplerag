import re
from typing import List
from langchain_core.documents import Document


class TextChunkingService:
    """Handles text chunking and splitting strategies."""

    def __init__(self, default_chunk_size: int = 1000, sentence_overlap: int = 2):
        self.default_chunk_size = default_chunk_size
        self.sentence_overlap = sentence_overlap

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences on boundary markers."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_by_sentences(self, text: str, target_chunk_size: int = None, sentence_overlap: int = None) -> List[Document]:
        """Chunk text by sentence boundaries with semantic overlap."""
        target_chunk_size = target_chunk_size or self.default_chunk_size
        sentence_overlap = sentence_overlap or self.sentence_overlap

        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > target_chunk_size and current_chunk:
                chunks.append(Document(page_content=' '.join(current_chunk)))
                current_chunk = current_chunk[-sentence_overlap:] if len(current_chunk) > sentence_overlap else current_chunk
                current_size = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(Document(page_content=' '.join(current_chunk)))

        return chunks
