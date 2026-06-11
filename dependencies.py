import os

from claude_service import ClaudeService
from gemini_service import GeminiService
from config.ai_config import ai_config
from ai_service import AIservice
from config.db_config import db_config
from rag.db_service import DBService
from config.embedding_config import embedding_config
from rag.embedding_service import EmbeddingService
from rag.ingest_data_service import IngestDataService
from rag.chunk_query_service import ChunkQueryService

_active_ai_service: AIservice = None
_db_service: DBService = None
_embedding_service: EmbeddingService = None
_ingest_data_service: IngestDataService = None
_chunk_query_service: ChunkQueryService = None

def get_ai_service() -> AIservice:

    global _active_ai_service

    if _active_ai_service is not None:
        return _active_ai_service
        
    elif ai_config.AI_PROVIDER == 'ANTHROPIC':
        if not ai_config.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key is not set in the environment variables.")
        _active_ai_service = ClaudeService(api_key=ai_config.ANTHROPIC_API_KEY, model=ai_config.ANTHROPIC_MODEL)

    elif ai_config.AI_PROVIDER == 'GEMINI':
        if not ai_config.GEMINI_API_KEY:
            raise ValueError("Gemini API key is not set in the environment variables.")
        _active_ai_service = GeminiService(api_key=ai_config.GEMINI_API_KEY, model=ai_config.GEMINI_MODEL)

    else:
        raise ValueError(f"Unsupported AI provider: {ai_config.AI_PROVIDER}")

    return _active_ai_service

def get_db_service() -> DBService:
    global _db_service

    if _db_service is None:
        _db_service = DBService(db_config)

    return _db_service

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    
    if _embedding_service is not None:
        return _embedding_service
    
    _embedding_service = EmbeddingService(
        embedding_model_api_key=embedding_config.EMBEDDING_MODEL_API_KEY,
        embedding_model=embedding_config.EMBEDDING_MODEL
    )
    return _embedding_service

def get_ingest_data_service() -> IngestDataService:
    global _ingest_data_service

    if _ingest_data_service is not None:
        return _ingest_data_service

    db_service = get_db_service()
    embedding_service = get_embedding_service()
    _ingest_data_service = IngestDataService(db_service=db_service, embedding_service=embedding_service)

    return _ingest_data_service

def get_chunk_query_service() -> ChunkQueryService:
    global _chunk_query_service

    if _chunk_query_service is not None:
        return _chunk_query_service

    db_service = get_db_service()
    embedding_service = get_embedding_service()
    ai_service = get_ai_service()
    _chunk_query_service = ChunkQueryService(db_service=db_service, embedding_service=embedding_service, ai_service=ai_service)

    return _chunk_query_service