from typing import List
from rag.db_service import DBService
from rag.embedding_service import EmbeddingService
from rag.get_chunks_from_text import get_chunks_from_text


class IngestDataService:
    def __init__(self, db_service: DBService, embedding_service: EmbeddingService, chunk_size: int = 200, chunk_overlap: int = 50):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest_text(self, text: str, source: str = None, metadata: dict = None) -> int:
        """Ingest text, chunk it, and store in database."""
        chunks = get_chunks_from_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        chunk_ids = []

        for idx, chunk in enumerate(chunks):
            chunk_metadata = metadata or {}
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["source"] = source

            chunk_embedding = await self.embedding_service.embed_content(chunk)
            print(f"Chunk embedding: {chunk_embedding[:5]}...{len(chunk_embedding)}")  # Print first

            chunk_id = await self.store_chunk(
                chunk_text=chunk,
                source=source,
                chunk_index=idx,
                metadata=chunk_metadata,
                embedding=chunk_embedding
            )
            chunk_ids.append(chunk_id)

        return len(chunk_ids)

    async def store_chunk(
        self,
        chunk_text: str,
        source: str = None,
        chunk_index: int = 0,
        metadata: dict = None,
        embedding: list = None
    ) -> int:
        """Store single chunk in database."""
        query = """
        INSERT INTO document_vectors (chunk_text, source, chunk_index, embedding)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """
        result = await self.db_service.execute(
            query,
            chunk_text,
            source,
            chunk_index,
            str(embedding)
        )
        return result

    async def ingest_file(self, file_path: str, metadata: dict = None) -> int:
        """Ingest text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        file_metadata = metadata or {}
        file_metadata["file_path"] = file_path

        return await self.ingest_text(text, source=file_path, metadata=file_metadata)

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

    async def search_chunks(self, search_text: str, limit: int = 2) -> List[dict]:
        
        # get text embedding
        search_embedding = await self.embedding_service.embed_content(search_text)
        # perform cosine similarity search using embeddings using postgres vector operations
        return await self.search_chunks_by_embedding(search_embedding, limit)

    
    # perform cosine similarity search using embeddings using postgres vector operations
    async def search_chunks_by_embedding(self, embedding: list, limit: int = 2) -> List[dict]:
        """Search chunks by embedding similarity."""
        query = """
        SELECT id, chunk_text, source, chunk_index
        FROM document_vectors
        ORDER BY (embedding <=> $1::vector) ASC
        LIMIT $2
        """

        result = await self.db_service.query(query, str(embedding), limit)
        print(f"Search results: {result}")

