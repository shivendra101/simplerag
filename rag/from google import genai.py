from google import genai
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


client = genai.Client(
    api_key=os.getenv("EMBEDDING_MODEL_API_KEY")
)

result = client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL"),
        contents="What is the meaning of life?"
)

print(result.embeddings)