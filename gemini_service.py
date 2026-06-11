from google import genai
from ai_service import AIservice

class GeminiService(AIservice):
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def ask_question(self, question: str, context: str = None) -> str:
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

        response = self.client.models.generate_content(
            model=self.model,
            input=[{"role": "user", "content": user_content}]
        )
        return response.candidates[0].content

    async def close(self):
        await self.client.close()