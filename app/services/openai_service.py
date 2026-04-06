from __future__ import annotations

from itertools import islice
from typing import Iterable

from openai import OpenAI

from app.config import Settings
from app.services.grounding import INSUFFICIENT_CONTEXT_MESSAGE, validate_grounded_answer
from app.types import SearchHit


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Add it to .env before indexing or asking questions.")
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self._batched(texts, batch_size=64):
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def answer_question(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return INSUFFICIENT_CONTEXT_MESSAGE

        context_blocks = []
        for index, hit in enumerate(hits, start=1):
            source_label = self._format_source_label(hit.metadata)
            context_blocks.append(f"[SOURCE {index}] {source_label}\n{hit.text}")

        prompt = "\n\n".join(context_blocks)
        response = self.client.chat.completions.create(
            model=self.settings.answer_model,
            temperature=self.settings.answer_temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ти внутрішній AI-асистент для бухгалтера компанії. "
                        "Відповідай тільки на основі наданого контексту. "
                        "Не використовуй зовнішні знання, не домислюй, не узагальнюй і не вигадуй відсутні значення. "
                        "Якщо хоч одна частина відповіді не підтверджується джерелами, напиши тільки: "
                        f"'{INSUFFICIENT_CONTEXT_MESSAGE}'. "
                        "Кожне речення з фактом повинно містити посилання на джерело у форматі [SOURCE n]. "
                        "Останній рядок повинен починатися з 'Джерела:' і перераховувати тільки використані [SOURCE n]."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Питання:\n{question}\n\n"
                        f"Контекст:\n{prompt}"
                    ),
                },
            ],
        )
        message = response.choices[0].message.content or ""
        normalized = message.strip()
        if not validate_grounded_answer(normalized, hits):
            return INSUFFICIENT_CONTEXT_MESSAGE
        return normalized

    def _format_source_label(self, metadata: dict[str, object]) -> str:
        details = [str(metadata.get("source_name", "unknown"))]
        if metadata.get("page"):
            details.append(f"сторінка {metadata['page']}")
        if metadata.get("sheet_name"):
            details.append(f"лист {metadata['sheet_name']}")
        if metadata.get("row_number"):
            details.append(f"рядок {metadata['row_number']}")
        return ", ".join(details)

    def _batched(self, values: list[str], batch_size: int) -> Iterable[list[str]]:
        iterator = iter(values)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch
