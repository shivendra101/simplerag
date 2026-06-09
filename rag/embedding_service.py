from google import genai
from google.genai import types


class EmbeddingService:
    def __init__(self, embedding_model_api_key: str, embedding_model: str):
        self.embedding_model_api_key = embedding_model_api_key
        self.embedding_model = embedding_model
        self.client = genai.Client(api_key=self.embedding_model_api_key)
    
    async def embed_content(self, content: str) -> list[float]:
        """Method to get embeddings for the given content using the specified embedding model."""
        
        embedding = self.client.models.embed_content(
            model=self.embedding_model,
            contents=content,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=1536)
        )

        return embedding.embeddings[0].values

    async def embed_multiple(self, texts: list) -> list:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_content(text)
            embeddings.append(embedding)
        return embeddings