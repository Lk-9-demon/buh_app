from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.services.parsers import DocumentParser
from app.services.registry import FileRegistry
from app.services.search_index import SearchIndex


class _FakeOpenAIService:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


class IndexCleanupTests(unittest.TestCase):
    def test_reindex_removes_deleted_files_from_registry_and_index(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            base_dir = Path(temp_dir)
            documents_dir = base_dir / "documents"
            storage_dir = base_dir / "storage"
            chroma_dir = storage_dir / "chroma"
            state_db_path = storage_dir / "state.db"

            documents_dir.mkdir(parents=True, exist_ok=True)
            storage_dir.mkdir(parents=True, exist_ok=True)
            chroma_dir.mkdir(parents=True, exist_ok=True)

            settings = Settings(
                documents_dir=documents_dir,
                storage_dir=storage_dir,
                chroma_dir=chroma_dir,
                state_db_path=state_db_path,
                allowed_extensions=(".txt",),
                index_schema_version="test_cleanup",
            )
            parser = DocumentParser(settings)
            registry = FileRegistry(state_db_path)
            search_index = SearchIndex(
                settings=settings,
                parser=parser,
                registry=registry,
                ollama_service=_FakeOpenAIService(),
            )

            first_file = documents_dir / "first.txt"
            second_file = documents_dir / "second.txt"
            first_file.write_text("Перший документ для тесту.", encoding="utf-8")
            second_file.write_text("Другий документ для тесту.", encoding="utf-8")

            initial_report = search_index.index_documents()
            self.assertEqual(initial_report.indexed, 2)
            self.assertEqual(initial_report.removed, 0)
            self.assertEqual(len(registry.list_records(limit=None)), 2)

            first_file.unlink()

            next_report = search_index.index_documents()
            self.assertEqual(next_report.removed, 1)
            self.assertIsNone(registry.get_record(first_file))
            self.assertEqual(len(registry.list_records(limit=None)), 1)


if __name__ == "__main__":
    unittest.main()
