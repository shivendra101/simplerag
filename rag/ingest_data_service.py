from typing import List
from rag.db_service import DBService
from rag.embedding_service import EmbeddingService
from rag.get_chunks_from_text import get_chunks_from_text

import fitz  # PyMuPDF engine
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class IngestDataService:
    def __init__(self, db_service: DBService, embedding_service: EmbeddingService, chunk_size: int = 200, chunk_overlap: int = 50):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def ingest_text(self, text: str, source: str = None, metadata: dict = None) -> int:

        # print(f"Ingesting text with source: {source} and metadata: {metadata}")
        # print(f"Text length: {text[:100]}...{len(text)}")  # Print first 100 characters and total length
        """Ingest text, chunk it, and store in database."""
        chunks = get_chunks_from_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        chunk_ids = []

        for idx, chunk in enumerate(chunks):
            chunk_metadata = metadata or {}
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["source"] = source

            chunk_embedding = await self.embedding_service.embed_content(chunk)

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
        """Ingest text from pdf file."""

        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]

        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        final_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        all_final_chunks = []
        chunk_counter = 0

        # Open the PDF document as a stream pointer
        with fitz.open(file_path) as doc:
            # Process page by page instead of all at once
            print(f"Total pages in document: {len(doc)}")
            for page_num in range(len(doc)):
                # Convert only ONE specific page to Markdown text
                page_md = pymupdf4llm.to_markdown(doc, pages=[page_num])
                
                print(f"page number {page_num + 1}")
                if not page_md.strip():
                    print(f"Page {page_num + 1} is blank, skipping.")
                    continue  # Skip blank pages
                    
                # Extract structural markdown blocks for this page
                page_structural_chunks = header_splitter.split_text(page_md)
                
                # Breakdown into smaller embedding-sized text windows
                page_final_docs = final_splitter.split_documents(page_structural_chunks)
                
                # Enrich metadata instantly with page numbers to prevent data loss
                for doc_chunk in page_final_docs:
                    doc_chunk.metadata["source"] = file_path
                    doc_chunk.metadata["page_number"] = page_num + 1
                    doc_chunk.metadata["chunk_index"] = f"{file_path}_p{page_num+1}_c{chunk_counter}"
                    
                    all_final_chunks.append(doc_chunk)
                    chunk_counter += 1
                    await self.ingest_text(doc_chunk.page_content, source=file_path, metadata=doc_chunk.metadata)

        return len(all_final_chunks)


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
        
        # get text embedding
        search_embedding = await self.embedding_service.embed_content(search_text)
        # perform cosine similarity search using embeddings using postgres vector operations
        return await self.search_chunks_by_embedding(search_embedding, limit)

    
    # perform cosine similarity search using embeddings using postgres vector operations
    async def search_chunks_by_embedding(self, embedding: list, limit: int = 10) -> List[dict]:
        """Search chunks by embedding similarity."""
        query = """
        SELECT id, chunk_text, source, chunk_index
        FROM document_vectors
        ORDER BY (embedding <=> $1::vector) ASC
        LIMIT $2
        """

        result = await self.db_service.query(query, str(embedding), limit)
        # get list of chunk_text from result
        all_chunks = []
        for row in result:
            all_chunks.append(row['chunk_text'])
            print(f"{row['chunk_text']}")
        
        return all_chunks

