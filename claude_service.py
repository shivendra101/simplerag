from anthropic import HUMAN_PROMPT, AI_PROMPT, AsyncAnthropic
from ai_service import AIservice

class ClaudeService(AIservice):
    def __init__(self, api_key: str, model: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def ask_question(self, question: str) -> str:
        message = await self.client.messages.create(
            model = self.model,
            max_tokens = 1024,
            messages = [
                {"role": "user", "content": question}
            ]
        )

        # print inout and output tokens for debugging
        print(f"Input tokens: {message.usage.input_tokens}")
        print(f"Output tokens: {message.usage.output_tokens}")

        return message.content[0].text

    async def close(self) -> None:
        await self.client.close()