from __future__ import annotations

import unittest

from app.services.grounding import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    filter_grounded_hits,
    validate_grounded_answer,
)
from app.types import SearchHit


def _hit(distance: float, source_name: str = "doc.pdf") -> SearchHit:
    return SearchHit(
        text="test chunk",
        metadata={"source_name": source_name},
        distance=distance,
        collection_name="doc_chunks",
    )


class GroundingTests(unittest.TestCase):
    def test_filter_grounded_hits_removes_far_results(self) -> None:
        hits = [_hit(0.21), _hit(0.28), _hit(0.51), _hit(0.91)]

        filtered = filter_grounded_hits(
            hits=hits,
            max_search_distance=0.65,
            max_distance_spread=0.12,
            max_context_chunks=8,
        )

        self.assertEqual([round(item.distance, 2) for item in filtered], [0.21, 0.28])

    def test_filter_grounded_hits_returns_empty_when_top_hit_is_too_far(self) -> None:
        filtered = filter_grounded_hits(
            hits=[_hit(0.82), _hit(0.9)],
            max_search_distance=0.65,
            max_distance_spread=0.12,
            max_context_chunks=8,
        )

        self.assertEqual(filtered, [])

    def test_validate_grounded_answer_requires_citations(self) -> None:
        valid = validate_grounded_answer(
            "Ціна пшениці 7800 грн/т.\nДжерела: [SOURCE 1]",
            [_hit(0.2)],
        )

        self.assertFalse(valid)

    def test_validate_grounded_answer_rejects_unknown_source_numbers(self) -> None:
        valid = validate_grounded_answer(
            "Ціна пшениці 7800 грн/т [SOURCE 2]\nДжерела: [SOURCE 2]",
            [_hit(0.2)],
        )

        self.assertFalse(valid)

    def test_validate_grounded_answer_accepts_grounded_output(self) -> None:
        valid = validate_grounded_answer(
            "Ціна пшениці у березні становить 7800 грн/т [SOURCE 1]\n"
            "Джерела: [SOURCE 1]",
            [_hit(0.2)],
        )

        self.assertTrue(valid)

    def test_validate_grounded_answer_accepts_insufficient_context_message(self) -> None:
        self.assertTrue(validate_grounded_answer(INSUFFICIENT_CONTEXT_MESSAGE, [_hit(0.2)]))


if __name__ == "__main__":
    unittest.main()
