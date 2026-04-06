from __future__ import annotations

import sqlite3
from pathlib import Path

from app.types import FileRecord


class FileRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    indexed_at TEXT,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    schema_version TEXT NOT NULL,
                    embedding_model TEXT NOT NULL
                )
                """
            )

    def get_record(self, path: Path) -> FileRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM indexed_files WHERE path = ?",
                (str(path),),
            ).fetchone()
        if row is None:
            return None
        return FileRecord(**dict(row))

    def list_records(self, limit: int = 100) -> list[FileRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM indexed_files
                ORDER BY COALESCE(indexed_at, modified_at) DESC, path ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [FileRecord(**dict(row)) for row in rows]

    def get_stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS total
                FROM indexed_files
                GROUP BY status
                """
            ).fetchall()
        stats = {"indexed": 0, "failed": 0, "pending": 0, "total": 0}
        for row in rows:
            status = row["status"]
            total = int(row["total"])
            stats[status] = total
            stats["total"] += total
        return stats

    def needs_indexing(
        self,
        path: Path,
        sha256: str,
        schema_version: str,
        embedding_model: str,
    ) -> bool:
        record = self.get_record(path)
        if record is None:
            return True
        if record.status != "indexed":
            return True
        if record.sha256 != sha256:
            return True
        if record.schema_version != schema_version:
            return True
        if record.embedding_model != embedding_model:
            return True
        return False

    def mark_indexed(
        self,
        path: Path,
        sha256: str,
        modified_at: str,
        indexed_at: str,
        chunk_count: int,
        schema_version: str,
        embedding_model: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO indexed_files (
                    path, sha256, modified_at, indexed_at, status,
                    chunk_count, last_error, schema_version, embedding_model
                )
                VALUES (?, ?, ?, ?, 'indexed', ?, NULL, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    sha256 = excluded.sha256,
                    modified_at = excluded.modified_at,
                    indexed_at = excluded.indexed_at,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count,
                    last_error = excluded.last_error,
                    schema_version = excluded.schema_version,
                    embedding_model = excluded.embedding_model
                """,
                (
                    str(path),
                    sha256,
                    modified_at,
                    indexed_at,
                    chunk_count,
                    schema_version,
                    embedding_model,
                ),
            )

    def mark_failed(
        self,
        path: Path,
        sha256: str,
        modified_at: str,
        error_message: str,
        schema_version: str,
        embedding_model: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO indexed_files (
                    path, sha256, modified_at, indexed_at, status,
                    chunk_count, last_error, schema_version, embedding_model
                )
                VALUES (?, ?, ?, NULL, 'failed', 0, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    sha256 = excluded.sha256,
                    modified_at = excluded.modified_at,
                    indexed_at = excluded.indexed_at,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count,
                    last_error = excluded.last_error,
                    schema_version = excluded.schema_version,
                    embedding_model = excluded.embedding_model
                """,
                (
                    str(path),
                    sha256,
                    modified_at,
                    error_message,
                    schema_version,
                    embedding_model,
                ),
            )
