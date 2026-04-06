from __future__ import annotations

import unittest
from pathlib import Path

from app.services.document_preview import (
    build_native_open_url,
    build_source_preview_url,
    render_answer_html,
)
from app.types import SearchHit


class DocumentPreviewTests(unittest.TestCase):
    def test_build_source_preview_url_includes_location_metadata(self) -> None:
        metadata = {
            "source_path": r"C:\docs\report.xlsx",
            "page": 3,
            "sheet_name": "Березень 2025",
            "row_number": 18,
        }

        result = build_source_preview_url(metadata)

        self.assertIn("/documents/preview?", result)
        self.assertIn("path=C%3A%5Cdocs%5Creport.xlsx", result)
        self.assertIn("page=3", result)
        self.assertIn("sheet_name=%D0%91%D0%B5%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%8C+2025", result)
        self.assertIn("row_number=18", result)

    def test_render_answer_html_wraps_source_reference_into_link(self) -> None:
        sources = [
            SearchHit(
                text="Ціна: 7800",
                metadata={
                    "source_path": r"C:\docs\prices.xlsx",
                    "source_name": "prices.xlsx",
                    "sheet_name": "Sheet1",
                    "row_number": 12,
                },
                distance=0.12,
                collection_name="excel_rows",
            )
        ]

        html = str(render_answer_html("Знайдено значення [SOURCE 1].", sources))

        self.assertIn('class="source-inline-link"', html)
        self.assertIn("/documents/preview?", html)
        self.assertIn("[SOURCE 1]</a>", html)

    def test_build_native_open_url_uses_excel_protocol_for_spreadsheets(self) -> None:
        result = build_native_open_url(Path(r"C:\docs\sheet.xlsx"))

        self.assertTrue(result.startswith("ms-excel:ofe|u|file:///"))


if __name__ == "__main__":
    unittest.main()
