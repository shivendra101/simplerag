from anthropic import HUMAN_PROMPT, AI_PROMPT, AsyncAnthropic
from ai_service import AIservice

class ClaudeService(AIservice):
    def __init__(self, api_key: str, model: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def ask_question(self, question: str, context: str = None) -> str:
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

        message = await self.client.messages.create(
            model = self.model,
            max_tokens = 1024,
            cache_control={ "type": "ephemeral" },
            system= [
                {
                "type": "text",
                "text": "You are a helpful assistant.",
                "cache_control": { "type": "ephemeral" }
                }
            ],
            messages = [
                {"role": "user", "content": user_content}
            ]
        )

        # print inout and output tokens for debugging
        print(f"Input tokens: {message.usage.input_tokens}")
        print(f"Output tokens: {message.usage.output_tokens}")
        print(f"Cache creation tokens: {getattr(message.usage, 'cache_creation_input_tokens', 0)}")
        print(f"Cache read tokens: {getattr(message.usage, 'cache_read_input_tokens', 0)}")
        print(f"Full usage: {message.usage}")
        print(f"Usage dict: {message.usage.__dict__}")

        return message.content[0].text

    async def close(self) -> None:
        await self.client.close()