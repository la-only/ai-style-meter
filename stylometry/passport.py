"""Style-passport construction, persistence, and z-normalized scoring."""

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stylometry.features import FEATURE_NAMES, extract_features
from stylometry.models import TextChunk


@dataclass(frozen=True)
class FeatureStats:
    """Population normalization parameters for one feature."""

    mean: float
    std: float


@dataclass(frozen=True)
class StylePassport:
    """Serializable author style profile."""

    author: str
    features: dict[str, FeatureStats]
    n_chunks: int
    chunk_words: int
    training_files: tuple[str, ...]
    excluded_files: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to the stable JSON representation."""

        return {
            "schema_version": self.schema_version,
            "author": self.author,
            "n_chunks": self.n_chunks,
            "chunk_words": self.chunk_words,
            "training_files": list(self.training_files),
            "excluded_files": list(self.excluded_files),
            "features": {
                name: {"mean": self.features[name].mean, "std": self.features[name].std}
                for name in FEATURE_NAMES
            },
        }


def build_style_passport(
    chunks: list[TextChunk],
    *,
    author: str,
    min_chunks: int = 100,
    chunk_words: int = 0,
    excluded_files: tuple[str, ...] = (),
) -> StylePassport:
    """Aggregate finite per-chunk features into population mean/std values."""

    if len(chunks) < min_chunks:
        raise ValueError(f"Style passport requires at least {min_chunks} chunks")
    vectors = [extract_features(chunk.text) for chunk in chunks]
    stats: dict[str, FeatureStats] = {}
    for name in FEATURE_NAMES:
        values = [vector[name] for vector in vectors]
        standard_deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        stats[name] = FeatureStats(
            mean=statistics.fmean(values),
            std=standard_deviation if standard_deviation > 1e-12 else 1.0,
        )
    return StylePassport(
        author=author,
        features=stats,
        n_chunks=len(chunks),
        chunk_words=chunk_words,
        training_files=tuple(sorted({chunk.source_file for chunk in chunks})),
        excluded_files=tuple(sorted(excluded_files)),
    )


def distance_to_passport(text: str, passport: StylePassport) -> float:
    """Return mean absolute z-score using only the passport normalization."""

    vector = extract_features(text)
    distance = statistics.fmean(
        abs((vector[name] - passport.features[name].mean) / passport.features[name].std)
        for name in FEATURE_NAMES
    )
    if not math.isfinite(distance):
        raise ValueError("Style distance must be finite")
    return distance


def save_style_passport(passport: StylePassport, path: Path) -> None:
    """Write strict JSON, refusing non-standard NaN/Infinity values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(passport.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_style_passport(path: Path) -> StylePassport:
    """Load and fully validate a passport before scoring untrusted text."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported style passport schema_version")
    feature_payload = payload.get("features", {})
    if tuple(feature_payload) != FEATURE_NAMES:
        raise ValueError("Style passport features do not match FEATURE_NAMES")
    features: dict[str, FeatureStats] = {}
    for name in FEATURE_NAMES:
        mean = float(feature_payload[name]["mean"])
        std = float(feature_payload[name]["std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization for feature {name}")
        features[name] = FeatureStats(mean=mean, std=std)
    n_chunks = int(payload["n_chunks"])
    if n_chunks < 100:
        raise ValueError("Style passport requires at least 100 chunks")
    return StylePassport(
        author=str(payload["author"]),
        features=features,
        n_chunks=n_chunks,
        chunk_words=int(payload["chunk_words"]),
        training_files=tuple(payload["training_files"]),
        excluded_files=tuple(payload["excluded_files"]),
    )
