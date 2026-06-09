from abc import ABC, abstractmethod

class AIservice(ABC):

    async def ask_question(self, question: str) -> str:
        """Abstract method to ask a question to the AI service."""
        pass

    async def close(self) -> None:
        """Abstract method to close any resources if needed."""
        pass