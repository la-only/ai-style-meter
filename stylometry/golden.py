"""Golden masking metadata and the mandatory human-review gate."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenExcerpt:
    """One masked excerpt; every listed source file is quarantined."""

    excerpt_id: str
    source_file: str
    original: str
    masked: str
    review: dict[str, Any] | None

    @property
    def reviewed(self) -> bool:
        """Only an explicitly approved, attributed human review opens the gate."""

        if self.review is None or self.review.get("status") != "approved":
            return False
        reviewer = self.review.get("reviewer")
        timestamp = self.review.get("reviewed_at")
        if not isinstance(reviewer, str) or not reviewer.strip():
            return False
        if not isinstance(timestamp, str) or "T" not in timestamp:
            return False
        try:
            parsed = datetime.fromisoformat(
                timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
            )
        except ValueError:
            return False
        return parsed.tzinfo is not None and parsed.utcoffset() is not None


def load_golden(path: Path | None) -> list[GoldenExcerpt]:
    """Read versioned JSONL and reject malformed or unsafe source references."""

    if path is None:
        return []
    excerpts: list[GoldenExcerpt] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            excerpt_id = str(payload["id"])
            source_file = str(payload["source_file"])
            source_path = Path(source_file)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise ValueError(f"Unsafe golden source_file at line {line_number}")
            if excerpt_id in seen_ids:
                raise ValueError(f"Duplicate golden id: {excerpt_id}")
            seen_ids.add(excerpt_id)
            review = payload.get("review")
            if review is not None and not isinstance(review, dict):
                raise ValueError(f"Invalid review metadata at line {line_number}")
            excerpts.append(
                GoldenExcerpt(
                    excerpt_id=excerpt_id,
                    source_file=source_file,
                    original=str(payload["original"]),
                    masked=str(payload["masked"]),
                    review=review,
                )
            )
    return excerpts
