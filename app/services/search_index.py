from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import chromadb

from app.config import Settings
from app.services.grounding import filter_grounded_hits
from app.services.ollama_service import OllamaService
from app.services.parsers import DocumentParser
from app.services.registry import FileRegistry
from app.types import IndexingReport, ParsedChunk, SearchHit


class SearchIndex:
    def __init__(
        self,
        settings: Settings,
        parser: DocumentParser,
        registry: FileRegistry,
        ollama_service: OllamaService,
    ) -> None:
        self.settings = settings
        self.parser = parser
        self.registry = registry
        self.ollama_service = ollama_service
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))

    def ensure_collections(self) -> None:
        self._get_collection("doc_chunks")
        self._get_collection("excel_rows")

    def index_documents(
        self,
        directory: Path | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> IndexingReport:
        report = IndexingReport()
        target_dir = directory or self.settings.documents_dir
        candidate_paths = [
            path
            for path in sorted(target_dir.rglob("*"))
            if path.is_file() and self.parser.supports(path)
        ]
        current_paths = set(candidate_paths)
        total_files = len(candidate_paths)

        self._notify_progress(
            progress_callback,
            total_files=total_files,
            processed_files=0,
            current_file=None,
            report=report,
        )

        tracked_paths = {
            Path(record.path)
            for record in self.registry.list_records(limit=None)
        }
        stale_paths = sorted(path for path in tracked_paths if path not in current_paths)
        for stale_path in stale_paths:
            self._delete_existing_entries(stale_path)
            self.registry.delete_record(stale_path)
            report.removed += 1

        processed_files = 0

        for path in candidate_paths:
            self._notify_progress(
                progress_callback,
                total_files=total_files,
                processed_files=processed_files,
                current_file=str(path),
                report=report,
            )

            sha256 = self._hash_file(path)
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()

            if not self.registry.needs_indexing(
                path=path,
                sha256=sha256,
                schema_version=self.settings.index_schema_version,
                embedding_model=self.settings.embedding_model,
            ):
                report.skipped += 1
                processed_files += 1
                self._notify_progress(
                    progress_callback,
                    total_files=total_files,
                    processed_files=processed_files,
                    current_file=str(path),
                    report=report,
                )
                continue

            try:
                parsed_chunks = self.parser.parse_file(path)
                if not parsed_chunks:
                    raise ValueError("Не вдалося витягнути текст або таблиці з файла.")

                self._replace_document(path, sha256, parsed_chunks)
                self.registry.mark_indexed(
                    path=path,
                    sha256=sha256,
                    modified_at=modified_at,
                    indexed_at=datetime.now(UTC).isoformat(),
                    chunk_count=len(parsed_chunks),
                    schema_version=self.settings.index_schema_version,
                    embedding_model=self.settings.embedding_model,
                )
                report.indexed += 1
            except Exception as exc:
                report.failed += 1
                error_message = str(exc)
                report.errors.append(f"{path.name}: {error_message}")
                self.registry.mark_failed(
                    path=path,
                    sha256=sha256,
                    modified_at=modified_at,
                    error_message=error_message,
                    schema_version=self.settings.index_schema_version,
                    embedding_model=self.settings.embedding_model,
                )
            processed_files += 1
            self._notify_progress(
                progress_callback,
                total_files=total_files,
                processed_files=processed_files,
                current_file=str(path),
                report=report,
            )

        return report

    def search(self, question: str) -> list[SearchHit]:
        if not question.strip():
            return []

        collections: list[tuple[str, Any]] = []
        for prefix in ("doc_chunks", "excel_rows"):
            collection = self._get_collection(prefix)
            if collection.count() > 0:
                collections.append((prefix, collection))

        if not collections:
            return []

        query_embedding = self.ollama_service.embed_texts([question])[0]
        results: list[SearchHit] = []

        for prefix, collection in collections:

            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=self.settings.top_k_per_collection,
            )

            documents = response.get("documents", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0]

            for document, metadata, distance in zip(documents, metadatas, distances):
                results.append(
                    SearchHit(
                        text=document,
                        metadata=metadata,
                        distance=float(distance),
                        collection_name=prefix,
                    )
                )

        return filter_grounded_hits(
            hits=results,
            max_search_distance=self.settings.max_search_distance,
            max_distance_spread=self.settings.max_distance_spread,
            max_context_chunks=self.settings.max_context_chunks,
        )

    def _replace_document(self, path: Path, sha256: str, chunks: list[ParsedChunk]) -> None:
        self._delete_existing_entries(path)

        texts = [chunk.text for chunk in chunks]
        embeddings = self.ollama_service.embed_texts(texts)
        grouped_payload: dict[str, dict[str, list[object]]] = {
            "doc_chunks": {"ids": [], "documents": [], "embeddings": [], "metadatas": []},
            "excel_rows": {"ids": [], "documents": [], "embeddings": [], "metadatas": []},
        }

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            prefix = "excel_rows" if path.suffix.lower() in {".xls", ".xlsx"} else "doc_chunks"
            metadata = self._build_metadata(path, sha256, index, chunk.metadata)
            chunk_id = f"{sha256}:{index}"

            grouped_payload[prefix]["ids"].append(chunk_id)
            grouped_payload[prefix]["documents"].append(chunk.text)
            grouped_payload[prefix]["embeddings"].append(embedding)
            grouped_payload[prefix]["metadatas"].append(metadata)

        for prefix, payload in grouped_payload.items():
            if not payload["ids"]:
                continue
            collection = self._get_collection(prefix)
            collection.upsert(
                ids=payload["ids"],
                documents=payload["documents"],
                embeddings=payload["embeddings"],
                metadatas=payload["metadatas"],
            )

    def _delete_existing_entries(self, path: Path) -> None:
        for prefix in ("doc_chunks", "excel_rows"):
            collection = self._get_collection(prefix)
            existing = collection.get(where={"source_path": str(path)}, include=[])
            ids = existing.get("ids", [])
            if ids:
                collection.delete(ids=ids)

    def _build_metadata(
        self,
        path: Path,
        sha256: str,
        index: int,
        extra_metadata: dict[str, object],
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "source_path": str(path),
            "source_name": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "doc_sha256": sha256,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
            "parser_version": self.settings.index_schema_version,
            "embedding_model": self.settings.embedding_model,
            "record_index": index,
        }
        metadata.update(extra_metadata)
        return metadata

    def _get_collection(self, prefix: str) -> Any:
        return self.client.get_or_create_collection(
            name=self._collection_name(prefix),
            metadata={"hnsw:space": "cosine"},
        )

    def _collection_name(self, prefix: str) -> str:
        sanitized_model = re.sub(r"[^a-zA-Z0-9]+", "_", self.settings.embedding_model).strip("_").lower()
        return f"{prefix}_{sanitized_model}_{self.settings.index_schema_version}"

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _notify_progress(
        self,
        progress_callback: Callable[[dict[str, object]], None] | None,
        total_files: int,
        processed_files: int,
        current_file: str | None,
        report: IndexingReport,
    ) -> None:
        if progress_callback is None:
            return

        progress_callback(
            {
                "total_files": total_files,
                "processed_files": processed_files,
                "current_file": current_file,
                "report": IndexingReport(
                    indexed=report.indexed,
                    skipped=report.skipped,
                    removed=report.removed,
                    failed=report.failed,
                    errors=list(report.errors),
                ),
            }
        )
