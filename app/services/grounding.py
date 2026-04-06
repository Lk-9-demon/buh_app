from __future__ import annotations

import re

from app.types import SearchHit


INSUFFICIENT_CONTEXT_MESSAGE = (
    "Не знайшов у документах достатньо підтвердженої інформації, "
    "щоб відповісти без припущень."
)

_CITATION_PATTERN = re.compile(r"\[SOURCE\s+(\d+)\]")
_INSUFFICIENT_CONTEXT_PATTERN = re.compile(
    r"(не\s+знайш|недостатньо|бракує|не\s+можу\s+підтвердити)",
    re.IGNORECASE,
)


def filter_grounded_hits(
    hits: list[SearchHit],
    max_search_distance: float,
    max_distance_spread: float,
    max_context_chunks: int,
) -> list[SearchHit]:
    if not hits:
        return []

    ordered = sorted(hits, key=lambda item: item.distance)
    best_distance = ordered[0].distance
    if best_distance > max_search_distance:
        return []

    allowed_max_distance = min(max_search_distance, best_distance + max_distance_spread)
    filtered = [hit for hit in ordered if hit.distance <= allowed_max_distance]
    return filtered[:max_context_chunks]


def validate_grounded_answer(answer: str, sources: list[SearchHit]) -> bool:
    normalized = answer.strip()
    if not normalized:
        return False

    if _INSUFFICIENT_CONTEXT_PATTERN.search(normalized):
        return True

    citations = [int(match.group(1)) for match in _CITATION_PATTERN.finditer(normalized)]
    if not citations:
        return False

    max_source_id = len(sources)
    if any(source_id < 1 or source_id > max_source_id for source_id in citations):
        return False

    content_lines = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("джерела:"):
            continue
        content_lines.append(line)

    if not content_lines:
        return False

    return all(_CITATION_PATTERN.search(line) for line in content_lines)
