from __future__ import annotations

import json
import re
from itertools import islice
from typing import Any, Iterable
from urllib import error, request

from app.config import Settings
from app.services.grounding import INSUFFICIENT_CONTEXT_MESSAGE, validate_grounded_answer
from app.types import SearchHit


class OllamaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self._batched(texts, batch_size=32):
            response = self._post_json(
                "/api/embed",
                {
                    "model": self.settings.embedding_model,
                    "input": batch,
                    "truncate": True,
                },
                timeout_seconds=180,
            )
            batch_embeddings = response.get("embeddings", [])
            if len(batch_embeddings) != len(batch):
                raise RuntimeError("Ollama повернула неповний набір embeddings.")
            embeddings.extend(
                [[float(value) for value in embedding] for embedding in batch_embeddings]
            )
        return embeddings

    def answer_question(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return INSUFFICIENT_CONTEXT_MESSAGE

        context_blocks = []
        for index, hit in enumerate(hits, start=1):
            source_label = self._format_source_label(hit.metadata)
            context_blocks.append(f"[SOURCE {index}] {source_label}\n{hit.text}")

        response = self._post_json(
            "/api/chat",
            {
                "model": self.settings.answer_model,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self.settings.answer_temperature,
                },
                "messages": [
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
                        "content": self._build_user_prompt(question, context_blocks),
                    },
                ],
            },
            timeout_seconds=240,
        )
        message = response.get("message", {}).get("content", "")
        normalized = self._normalize_answer(message)
        if not validate_grounded_answer(normalized, hits):
            return INSUFFICIENT_CONTEXT_MESSAGE
        return normalized

    def get_status(self) -> dict[str, object]:
        try:
            models = self.list_models()
        except RuntimeError as exc:
            return {"ready": False, "message": str(exc), "missing_models": []}

        missing_models = [
            model_name
            for model_name in (self.settings.answer_model, self.settings.embedding_model)
            if model_name not in models
        ]
        if missing_models:
            return {
                "ready": False,
                "message": (
                    "В Ollama не вистачає моделей: "
                    + ", ".join(missing_models)
                    + ". Завантаж їх командою ollama pull."
                ),
                "missing_models": missing_models,
            }

        return {"ready": True, "message": None, "missing_models": []}

    def list_models(self) -> set[str]:
        response = self._get_json("/api/tags", timeout_seconds=10)
        models = response.get("models", [])
        names = {
            str(item.get("name") or item.get("model") or "").strip()
            for item in models
        }
        return {name for name in names if name}

    def _build_user_prompt(self, question: str, context_blocks: list[str]) -> str:
        context = "\n\n".join(context_blocks)
        return (
            f"Питання:\n{question}\n\n"
            f"Контекст:\n{context}"
        )

    def _normalize_answer(self, message: str) -> str:
        lines = []
        for raw_line in message.splitlines():
            line = raw_line.replace("**", "").strip()
            if not line:
                continue
            if line.lower().rstrip(":") in {"відповідь", "answer"}:
                continue
            if line.lower().startswith("джерела:"):
                line = self._normalize_sources_line(line)
            lines.append(line)

        normalized = "\n".join(lines).strip()
        if not normalized:
            return ""

        cited_source_ids = [int(match.group(1)) for match in re.finditer(r"\[SOURCE\s+(\d+)\]", normalized)]
        if cited_source_ids and not any(line.lower().startswith("джерела:") for line in lines):
            unique_source_ids = []
            seen: set[int] = set()
            for source_id in cited_source_ids:
                if source_id in seen:
                    continue
                seen.add(source_id)
                unique_source_ids.append(source_id)
            citations = ", ".join(f"[SOURCE {source_id}]" for source_id in unique_source_ids)
            normalized = f"{normalized}\nДжерела: {citations}"

        return normalized

    def _normalize_sources_line(self, line: str) -> str:
        source_ids = []
        seen: set[int] = set()
        for match in re.finditer(r"(?:\[)?SOURCE\s+(\d+)(?:\])?", line, flags=re.IGNORECASE):
            source_id = int(match.group(1))
            if source_id in seen:
                continue
            seen.add(source_id)
            source_ids.append(source_id)

        if not source_ids:
            return line

        citations = ", ".join(f"[SOURCE {source_id}]" for source_id in source_ids)
        return f"Джерела: {citations}"

    def _format_source_label(self, metadata: dict[str, object]) -> str:
        details = [str(metadata.get("source_name", "unknown"))]
        if metadata.get("page"):
            details.append(f"сторінка {metadata['page']}")
        if metadata.get("sheet_name"):
            details.append(f"лист {metadata['sheet_name']}")
        if metadata.get("row_number"):
            details.append(f"рядок {metadata['row_number']}")
        return ", ".join(details)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        body: bytes | None = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        base_url = self.settings.ollama_base_url.rstrip("/")
        http_request = request.Request(
            url=f"{base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"Ollama повернула помилку {exc.code} для {path}: {details or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Не вдалося підключитися до Ollama за адресою {self.settings.ollama_base_url}."
            ) from exc

        if not raw:
            return {}
        return json.loads(raw)

    def _get_json(self, path: str, timeout_seconds: int = 30) -> dict[str, Any]:
        return self._request("GET", path, payload=None, timeout_seconds=timeout_seconds)

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        return self._request("POST", path, payload=payload, timeout_seconds=timeout_seconds)

    def _batched(self, values: list[str], batch_size: int) -> Iterable[list[str]]:
        iterator = iter(values)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch
