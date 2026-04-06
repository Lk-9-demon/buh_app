from __future__ import annotations

import unittest

from app.config import Settings
from app.services.ollama_service import OllamaService


class _ReadyOllamaService(OllamaService):
    def __init__(self, settings: Settings, models: set[str], error_message: str | None = None) -> None:
        super().__init__(settings)
        self._models = models
        self._error_message = error_message

    def list_models(self) -> set[str]:
        if self._error_message is not None:
            raise RuntimeError(self._error_message)
        return self._models


class OllamaServiceTests(unittest.TestCase):
    def test_status_is_ready_when_required_models_are_available(self) -> None:
        settings = Settings()
        service = _ReadyOllamaService(
            settings=settings,
            models={settings.answer_model, settings.embedding_model},
        )

        status = service.get_status()

        self.assertTrue(status["ready"])
        self.assertEqual(status["missing_models"], [])

    def test_status_reports_missing_models(self) -> None:
        settings = Settings()
        service = _ReadyOllamaService(
            settings=settings,
            models={settings.answer_model},
        )

        status = service.get_status()

        self.assertFalse(status["ready"])
        self.assertIn(settings.embedding_model, status["missing_models"])

    def test_status_reports_connection_errors(self) -> None:
        service = _ReadyOllamaService(
            settings=Settings(),
            models=set(),
            error_message="Не вдалося підключитися до Ollama.",
        )

        status = service.get_status()

        self.assertFalse(status["ready"])
        self.assertEqual(status["message"], "Не вдалося підключитися до Ollama.")

    def test_normalize_answer_appends_sources_line(self) -> None:
        service = _ReadyOllamaService(settings=Settings(), models=set())

        normalized = service._normalize_answer("Ціна 7800 грн/т [SOURCE 1].")

        self.assertEqual(
            normalized,
            "Ціна 7800 грн/т [SOURCE 1].\nДжерела: [SOURCE 1]",
        )

    def test_normalize_answer_removes_heading_only_line(self) -> None:
        service = _ReadyOllamaService(settings=Settings(), models=set())

        normalized = service._normalize_answer("Відповідь:\nЦіна 7800 грн/т [SOURCE 1].")

        self.assertEqual(
            normalized,
            "Ціна 7800 грн/т [SOURCE 1].\nДжерела: [SOURCE 1]",
        )

    def test_normalize_answer_fixes_existing_sources_line(self) -> None:
        service = _ReadyOllamaService(settings=Settings(), models=set())

        normalized = service._normalize_answer(
            "Ціна 7800 грн/т [SOURCE 1].\nДжерела: SOURCE 1"
        )

        self.assertEqual(
            normalized,
            "Ціна 7800 грн/т [SOURCE 1].\nДжерела: [SOURCE 1]",
        )


if __name__ == "__main__":
    unittest.main()
