from __future__ import annotations

import time
import unittest

from app.services.indexing_manager import IndexingManager
from app.types import IndexingReport


class _FakeSearchIndex:
    def index_documents(self, directory=None, progress_callback=None) -> IndexingReport:
        report = IndexingReport(indexed=1, skipped=2, removed=0, failed=0)
        if progress_callback is not None:
            progress_callback(
                {
                    "total_files": 3,
                    "processed_files": 1,
                    "current_file": "demo.pdf",
                    "report": IndexingReport(indexed=1, skipped=0, removed=0, failed=0),
                }
            )
        return report


class IndexingManagerTests(unittest.TestCase):
    def test_start_runs_in_background_and_updates_status(self) -> None:
        manager = IndexingManager(_FakeSearchIndex())

        started = manager.start()

        self.assertTrue(started)

        deadline = time.time() + 2
        while manager.get_status().running and time.time() < deadline:
            time.sleep(0.01)

        status = manager.get_status()
        self.assertFalse(status.running)
        self.assertEqual(status.total_files, 3)
        self.assertEqual(status.processed_files, 3)
        self.assertEqual(status.report.indexed, 1)
        self.assertEqual(status.report.skipped, 2)

    def test_start_returns_false_when_job_is_already_running(self) -> None:
        class _SlowSearchIndex:
            def index_documents(self, directory=None, progress_callback=None) -> IndexingReport:
                time.sleep(0.2)
                return IndexingReport()

        manager = IndexingManager(_SlowSearchIndex())

        self.assertTrue(manager.start())
        self.assertFalse(manager.start())


if __name__ == "__main__":
    unittest.main()
