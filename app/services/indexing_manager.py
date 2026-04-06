from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread

from app.services.search_index import SearchIndex
from app.types import IndexingReport, IndexingStatus


class IndexingManager:
    def __init__(self, search_index: SearchIndex) -> None:
        self.search_index = search_index
        self._lock = Lock()
        self._status = IndexingStatus()
        self._worker: Thread | None = None

    def start(self, directory: Path | None = None) -> bool:
        with self._lock:
            if self._status.running:
                return False

            self._status = IndexingStatus(
                running=True,
                started_at=datetime.now(UTC).isoformat(),
                finished_at=None,
                total_files=0,
                processed_files=0,
                current_file=None,
                message="Індексація документів триває у фоні.",
                report=IndexingReport(),
            )
            self._worker = Thread(
                target=self._run_indexing,
                args=(directory,),
                daemon=True,
                name="buh-chat-indexing",
            )
            self._worker.start()
            return True

    def get_status(self) -> IndexingStatus:
        with self._lock:
            report = self._status.report
            return IndexingStatus(
                running=self._status.running,
                started_at=self._status.started_at,
                finished_at=self._status.finished_at,
                total_files=self._status.total_files,
                processed_files=self._status.processed_files,
                current_file=self._status.current_file,
                message=self._status.message,
                report=IndexingReport(
                    indexed=report.indexed,
                    skipped=report.skipped,
                    removed=report.removed,
                    failed=report.failed,
                    errors=list(report.errors),
                ),
            )

    def _run_indexing(self, directory: Path | None) -> None:
        try:
            report = self.search_index.index_documents(
                directory=directory,
                progress_callback=self._handle_progress,
            )
        except Exception as exc:
            with self._lock:
                self._status.running = False
                self._status.finished_at = datetime.now(UTC).isoformat()
                self._status.current_file = None
                self._status.message = f"Індексація зупинилась через помилку: {exc}"
                self._status.report.failed += 1
            return

        with self._lock:
            self._status.running = False
            self._status.finished_at = datetime.now(UTC).isoformat()
            self._status.current_file = None
            self._status.processed_files = self._status.total_files
            self._status.report = IndexingReport(
                indexed=report.indexed,
                skipped=report.skipped,
                removed=report.removed,
                failed=report.failed,
                errors=list(report.errors),
            )
            self._status.message = (
                f"Індексація завершена. Нових/оновлених: {report.indexed}, "
                f"пропущено: {report.skipped}, видалено з індексу: {report.removed}, помилок: {report.failed}."
            )

    def _handle_progress(self, payload: dict[str, object]) -> None:
        with self._lock:
            self._status.total_files = int(payload.get("total_files", 0))
            self._status.processed_files = int(payload.get("processed_files", 0))
            self._status.current_file = payload.get("current_file") or None
            incoming_report = payload.get("report")
            if isinstance(incoming_report, IndexingReport):
                self._status.report = IndexingReport(
                    indexed=incoming_report.indexed,
                    skipped=incoming_report.skipped,
                    removed=incoming_report.removed,
                    failed=incoming_report.failed,
                    errors=list(incoming_report.errors),
                )
