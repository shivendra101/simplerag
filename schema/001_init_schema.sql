-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Document vectors table for RAG system
CREATE TABLE IF NOT EXISTS document_vectors (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),
    source VARCHAR(255),
    chunk_index INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_document_vectors_embedding
ON document_vectors USING hnsw (embedding vector_cosine_ops);

-- Index for source lookups
CREATE INDEX IF NOT EXISTS idx_document_vectors_source
ON document_vectors(source);

-- Index for created_at range queries
CREATE INDEX IF NOT EXISTS idx_document_vectors_created_at
ON document_vectors(created_at DESC);
