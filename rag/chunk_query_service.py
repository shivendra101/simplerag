from typing import List
from rag.db_service import DBService
from rag.embedding_service import EmbeddingService


class ChunkQueryService:
    """Handles querying, searching, and deleting document chunks."""

    def __init__(self, db_service: DBService, embedding_service: EmbeddingService, ai_service=None):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.ai_service = ai_service

    async def get_chunks(self, source: str = None, limit: int = 100) -> List[dict]:
        """Retrieve chunks from database."""
        if source:
            query = "SELECT id, chunk_text, source, chunk_index, embedding FROM document_vectors WHERE source = $1 LIMIT $2"
            return await self.db_service.query(query, source, limit)
        else:
            query = "SELECT id, chunk_text, source, chunk_index, embedding FROM document_vectors LIMIT $1"
            return await self.db_service.query(query, limit)

    async def delete_chunks(self, source: str) -> int:
        """Delete all chunks from a source."""
        query = "DELETE FROM document_vectors WHERE source = $1"
        result = await self.db_service.execute(query, source)
        return result

    async def search_chunks(self, search_text: str, limit: int = 10) -> List[dict]:
        """Search chunks by text embedding similarity."""
        search_embedding = await self.embedding_service.embed_content(search_text)
        return await self.search_chunks_by_embedding(search_embedding, limit)

    async def search_chunks_by_embedding(self, embedding: list, limit: int = 10) -> List[dict]:
        """Search chunks by embedding similarity using cosine distance."""
        query = """
        SELECT id, chunk_text, source, chunk_index
        FROM document_vectors
        ORDER BY (embedding <=> $1::vector) ASC
        LIMIT $2
        """

        result = await self.db_service.query(query, str(embedding), limit)
        all_chunks = []
        for row in result:
            all_chunks.append(row['chunk_text'])
            print(f"{row['chunk_text']}")

        return all_chunks

    async def get_answer_to_query(self, question: str, limit: int = 5) -> dict:
        """RAG query: search relevant chunks and generate answer using AI service."""
        if not self.ai_service:
            raise ValueError("AI service not initialized")

        relevant_chunks = await self.search_chunks(question, limit=limit)

        context = "\n\n".join(relevant_chunks)

        answer = await self.ai_service.ask_question(question, context=context)

        return {
            "question": question,
            "answer": answer,
            "context_chunks_count": len(relevant_chunks)
        }
     