from __future__ import annotations

from app.services.openai_service import OpenAIService
from app.services.search_index import SearchIndex
from app.types import ChatAnswer


class ChatService:
    def __init__(self, search_index: SearchIndex, openai_service: OpenAIService) -> None:
        self.search_index = search_index
        self.openai_service = openai_service

    def answer_question(self, question: str) -> ChatAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            return ChatAnswer(
                question=question,
                answer="Постав питання, і я спробую знайти відповідь у документах.",
                sources=[],
            )

        hits = self.search_index.search(normalized_question)
        answer = self.openai_service.answer_question(normalized_question, hits)
        return ChatAnswer(question=normalized_question, answer=answer, sources=hits)
