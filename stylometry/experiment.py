"""Leakage-resistant held-out experiment and threshold governance."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from stylometry.golden import GoldenExcerpt, load_golden
from stylometry.models import Document, TextChunk, chunk_documents, load_documents
from stylometry.passport import (
    StylePassport,
    build_style_passport,
    distance_to_passport,
    save_style_passport,
)

DEFAULT_BASELINE_REPORT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "stylometry"
    / "baseline"
    / "experiment.json"
)


@dataclass(frozen=True)
class Thresholds:
    """Acceptance thresholds fixed before inspecting held-out results."""

    auc_min: float = 0.90
    mannwhitney_p_max: float = 0.01


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible experiment parameters."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    report_version: str = "1.0.0"
    chunk_words: int = 200
    min_chunks: int = 100
    held_out_fraction: float = 0.25
    random_state: int = 42
    tuning_split: str = "train"
    max_char_features: int = 50_000


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as error:
        raise ValueError(f"Invalid report version: {version}") from error


def validate_threshold_change(
    current: Thresholds,
    report_version: str,
    baseline_report: Path | None,
    *,
    tuning_split: str,
) -> None:
    """Require a version bump and forbid held-out threshold tuning."""

    if tuning_split == "held_out":
        raise ValueError("held_out cannot be used for threshold tuning")
    if baseline_report is None:
        if DEFAULT_BASELINE_REPORT.is_file():
            baseline_report = DEFAULT_BASELINE_REPORT
        elif current != Thresholds():
            raise ValueError("A baseline report is required for custom thresholds")
        else:
            return
    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
    baseline_thresholds = baseline["thresholds"]
    thresholds_changed = any(
        not math.isclose(
            getattr(current, name),
            float(baseline_thresholds[name]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for name in ("auc_min", "mannwhitney_p_max")
    )
    if not thresholds_changed:
        return
    if _version_tuple(report_version) <= _version_tuple(
        str(baseline["report_version"])
    ):
        raise ValueError("A threshold change requires a greater report version")


def split_documents(
    documents: list[Document], held_out_fraction: float
) -> tuple[list[Document], list[Document]]:
    """Hold out the last whole files, never individual chunks."""

    if len(documents) < 2:
        raise ValueError("Each corpus needs at least two files for a file-level split")
    if not 0 < held_out_fraction < 1:
        raise ValueError("held_out_fraction must be between zero and one")
    held_out_count = max(1, math.ceil(len(documents) * held_out_fraction))
    if held_out_count >= len(documents):
        raise ValueError("held_out split would leave no training files")
    split_at = len(documents) - held_out_count
    return documents[:split_at], documents[split_at:]


def _without_files(chunks: list[TextChunk], excluded: set[str]) -> list[TextChunk]:
    return [chunk for chunk in chunks if chunk.source_file not in excluded]


def _file_names(documents: list[Document]) -> list[str]:
    return [document.source_file for document in documents]


def _mean_distances(chunks: list[TextChunk], passport: StylePassport) -> list[float]:
    return [distance_to_passport(chunk.text, passport) for chunk in chunks]


def _drift_payload(
    documents: list[Document], passport: StylePassport, chunk_words: int
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for document in documents:
        chunks = chunk_documents([document], chunk_words=chunk_words)
        distances = _mean_distances(chunks, passport)
        if not distances:
            raise ValueError(f"File has no usable chunks: {document.source_file}")
        files.append(
            {
                "source_file": document.source_file,
                "n_chunks": len(chunks),
                "mean_distance": sum(distances) / len(distances),
            }
        )
    return {"schema_version": "1.0", "author": passport.author, "files": files}


def _golden_payload(
    excerpts: list[GoldenExcerpt], passport: StylePassport
) -> dict[str, Any]:
    reviewed = [excerpt for excerpt in excerpts if excerpt.reviewed]
    unreviewed = [excerpt for excerpt in excerpts if not excerpt.reviewed]
    return {
        "reviewed_ids": [excerpt.excerpt_id for excerpt in reviewed],
        "excluded_unreviewed_ids": [excerpt.excerpt_id for excerpt in unreviewed],
        "metrics": [
            {
                "id": excerpt.excerpt_id,
                "source_file": excerpt.source_file,
                "masked_distance": distance_to_passport(excerpt.masked, passport),
                "original_distance": distance_to_passport(excerpt.original, passport),
            }
            for excerpt in reviewed
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    *,
    author_dir: Path,
    other_dirs: list[Path],
    output_dir: Path,
    golden_path: Path | None = None,
    config: ExperimentConfig | None = None,
    baseline_report: Path | None = None,
) -> dict[str, Any]:
    """Train on file-disjoint data and evaluate frozen style metrics."""

    config = config or ExperimentConfig()
    validate_threshold_change(
        config.thresholds,
        config.report_version,
        baseline_report,
        tuning_split=config.tuning_split,
    )
    if not other_dirs:
        raise ValueError("At least one foreign corpus is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    author_documents = load_documents(author_dir)
    other_corpora = {
        directory.name: load_documents(directory) for directory in other_dirs
    }
    if len(other_corpora) != len(other_dirs):
        raise ValueError("Foreign corpus directory names must be unique")
    author_train_docs, author_held_out_docs = split_documents(
        author_documents, config.held_out_fraction
    )
    other_splits = {
        name: split_documents(documents, config.held_out_fraction)
        for name, documents in other_corpora.items()
    }

    golden = load_golden(golden_path)
    excluded_files = {excerpt.source_file for excerpt in golden}
    known_author_files = set(_file_names(author_documents))
    unknown_exclusions = excluded_files - known_author_files
    if unknown_exclusions:
        raise ValueError(
            f"Golden source files absent from author corpus: {unknown_exclusions}"
        )

    author_train_chunks = _without_files(
        chunk_documents(author_train_docs, config.chunk_words), excluded_files
    )
    passport = build_style_passport(
        author_train_chunks,
        author=author_dir.name,
        min_chunks=config.min_chunks,
        chunk_words=config.chunk_words,
        excluded_files=tuple(sorted(excluded_files)),
    )
    save_style_passport(passport, output_dir / "style.json")

    other_train_docs = [
        document for train, _ in other_splits.values() for document in train
    ]
    other_train_chunks = chunk_documents(other_train_docs, config.chunk_words)
    other_held_out_docs = [
        document for _, held_out in other_splits.values() for document in held_out
    ]
    discriminator_train_chunks = author_train_chunks + other_train_chunks
    author_held_out_chunks = chunk_documents(author_held_out_docs, config.chunk_words)
    other_held_out_chunks = chunk_documents(other_held_out_docs, config.chunk_words)
    evaluation_chunks = author_held_out_chunks + other_held_out_chunks
    if not discriminator_train_chunks or not evaluation_chunks:
        raise ValueError("Discriminator requires non-empty train and held-out chunks")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=config.max_char_features,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(
        [chunk.text for chunk in discriminator_train_chunks]
    )
    evaluation_matrix = vectorizer.transform(
        [chunk.text for chunk in evaluation_chunks]
    )
    train_labels = [1] * len(author_train_chunks) + [0] * len(other_train_chunks)
    evaluation_labels = [1] * len(author_held_out_chunks) + [0] * len(
        other_held_out_chunks
    )
    classifier = LogisticRegression(max_iter=1_000, random_state=config.random_state)
    classifier.fit(train_matrix, train_labels)
    probabilities = classifier.predict_proba(evaluation_matrix)[:, 1]
    auc = float(roc_auc_score(evaluation_labels, probabilities))

    author_distances = _mean_distances(author_held_out_chunks, passport)
    other_distances = _mean_distances(other_held_out_chunks, passport)
    if not author_distances or not other_distances:
        raise ValueError("Mann-Whitney requires non-empty held-out distance samples")
    mannwhitney = mannwhitneyu(author_distances, other_distances, alternative="less")

    drift = _drift_payload(author_documents, passport, config.chunk_words)
    _write_json(output_dir / "drift.json", drift)
    author_train_files = _file_names(author_train_docs)
    author_held_out_files = _file_names(author_held_out_docs)
    train_files = [chunk.source_file for chunk in discriminator_train_chunks]
    held_out_files = [chunk.source_file for chunk in evaluation_chunks]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "report_version": config.report_version,
        "author": author_dir.name,
        "thresholds": asdict(config.thresholds),
        "tuning_split": config.tuning_split,
        "split": {
            "method": "whole_files_last_sorted",
            "held_out_fraction": config.held_out_fraction,
            "author": {
                "all": _file_names(author_documents),
                "train": author_train_files,
                "held_out": author_held_out_files,
            },
            "others": {
                name: {"train": _file_names(train), "held_out": _file_names(held_out)}
                for name, (train, held_out) in other_splits.items()
            },
        },
        "passport": {
            "n_chunks": passport.n_chunks,
            "training_files": list(passport.training_files),
            "excluded_files": list(passport.excluded_files),
        },
        "discriminator": {
            "model": "LogisticRegression",
            "analyzer": "char",
            "ngram_range": [3, 5],
            "auc": auc,
            "train_files": sorted(set(train_files)),
            "held_out_files": sorted(set(held_out_files)),
            "n_train_chunks": len(discriminator_train_chunks),
            "n_held_out_chunks": len(evaluation_chunks),
        },
        "style_distance": {
            "mannwhitney": {
                "alternative": "less",
                "statistic": float(mannwhitney.statistic),
                "p_value": float(mannwhitney.pvalue),
            },
            "n_author_chunks": len(author_distances),
            "n_other_chunks": len(other_distances),
        },
        "golden": _golden_payload(golden, passport),
        "artifacts": {"style": "style.json", "drift": "drift.json"},
    }
    report["acceptance"] = {
        "auc_passed": auc >= config.thresholds.auc_min,
        "mannwhitney_passed": (
            float(mannwhitney.pvalue) < config.thresholds.mannwhitney_p_max
        ),
        "unreviewed_golden_excluded": all(
            metric["id"] not in report["golden"]["excluded_unreviewed_ids"]
            for metric in report["golden"]["metrics"]
        ),
    }
    report["passed"] = all(report["acceptance"].values())
    _write_json(output_dir / "experiment.json", report)
    return report
