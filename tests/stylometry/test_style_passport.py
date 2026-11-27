"""Acceptance tests for TASK-015, written before the implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest


def _author_sentence(file_index: int, sentence_index: int) -> str:
    return (
        f"Тихий герой{file_index} медленно смотрел на старый сад, "
        f"и тени{sentence_index} ложились между светлых деревьев. "
    )


def _foreign_sentence(file_index: int, sentence_index: int) -> str:
    return (
        f"Вдруг человек{file_index} стремительно воскликнул: "
        f"«Немедленно{sentence_index}, решительно, бесповоротно!» "
    )


def _technical_sentence(file_index: int, sentence_index: int) -> str:
    return (
        f"Модуль{file_index} выполняет проверку{sentence_index} входных данных; "
        "результат записывается в таблицу после завершения операции. "
    )


def _make_corpus(
    root: Path,
    prefix: str,
    sentence_factory: Any,
    *,
    files: int = 5,
    chunks_per_file: int = 8,
    chunk_words: int = 20,
) -> Path:
    directory = root / prefix
    directory.mkdir(parents=True)
    for file_index in range(files):
        text_parts: list[str] = []
        for sentence_index in range(chunks_per_file * 3):
            text_parts.append(sentence_factory(file_index, sentence_index))
        path = directory / f"{prefix}_{file_index}.txt"
        path.write_text("".join(text_parts), encoding="utf-8")
    return directory


@pytest.fixture
def synthetic_experiment(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    from stylometry.experiment import ExperimentConfig, run_experiment

    author_dir = _make_corpus(tmp_path, "author", _author_sentence)
    other_dirs = [
        _make_corpus(tmp_path, "foreign", _foreign_sentence),
        _make_corpus(tmp_path, "technical", _technical_sentence),
    ]
    golden_path = tmp_path / "golden.jsonl"
    golden_entries = [
        {
            "id": "reviewed",
            "source_file": "author_1.txt",
            "original": "Тихий герой смотрел на сад.",
            "masked": "Тихий герой [MASK] на сад.",
            "review": {
                "reviewer": "human-reviewer",
                "status": "approved",
                "reviewed_at": "2026-11-27T00:00:00Z",
            },
        },
        {
            "id": "not-reviewed",
            "source_file": "author_2.txt",
            "original": "Тени ложились между деревьев.",
            "masked": "Тени [MASK] между деревьев.",
        },
    ]
    golden_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in golden_entries),
        encoding="utf-8",
    )
    output_dir = tmp_path / "result"
    report = run_experiment(
        author_dir=author_dir,
        other_dirs=other_dirs,
        output_dir=output_dir,
        golden_path=golden_path,
        config=ExperimentConfig(chunk_words=20, min_chunks=2),
    )
    return report, output_dir


def test_style_card_schema_and_completeness(tmp_path: Path) -> None:
    from stylometry.features import FEATURE_NAMES
    from stylometry.models import TextChunk
    from stylometry.passport import build_style_passport, save_style_passport

    text = _author_sentence(0, 0) * 5
    chunks = [
        TextChunk(text=text, source_file="author.txt", chunk_index=i)
        for i in range(100)
    ]
    passport = build_style_passport(chunks, author="test-author", min_chunks=100)
    output = tmp_path / "style.json"
    save_style_passport(passport, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["author"] == "test-author"
    assert payload["n_chunks"] >= 100
    assert tuple(payload["features"].keys()) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert math.isfinite(payload["features"][name]["mean"])
        assert math.isfinite(payload["features"][name]["std"])
        assert payload["features"][name]["std"] > 0


def test_distance_is_z_normalized() -> None:
    from stylometry.models import TextChunk
    from stylometry.passport import build_style_passport, distance_to_passport

    text = _author_sentence(0, 0) * 5
    chunks = [
        TextChunk(text=text, source_file="author.txt", chunk_index=i)
        for i in range(100)
    ]
    passport = build_style_passport(chunks, author="test-author")
    foreign_text = _foreign_sentence(9, 9) * 10

    assert distance_to_passport(text, passport) == pytest.approx(0.0, abs=1e-12)
    assert distance_to_passport(foreign_text, passport) > 1.0

    from stylometry.features import FEATURE_NAMES, extract_features

    foreign_vector = extract_features(foreign_text)
    expected = sum(
        abs(
            (foreign_vector[name] - passport.features[name].mean)
            / passport.features[name].std
        )
        for name in FEATURE_NAMES
    ) / len(FEATURE_NAMES)
    assert distance_to_passport(foreign_text, passport) == pytest.approx(expected)


def test_discriminator_auc_threshold(
    synthetic_experiment: tuple[dict[str, Any], Path],
) -> None:
    report, _ = synthetic_experiment
    discriminator = report["discriminator"]
    split = report["split"]

    assert discriminator["auc"] >= report["thresholds"]["auc_min"]
    assert discriminator["analyzer"] == "char"
    assert discriminator["ngram_range"] == [3, 5]
    assert discriminator["model"] == "LogisticRegression"
    assert set(split["author"]["train"]).isdisjoint(split["author"]["held_out"])
    assert set(discriminator["train_files"]).isdisjoint(discriminator["held_out_files"])


def test_styledist_mannwhitney(
    synthetic_experiment: tuple[dict[str, Any], Path],
) -> None:
    report, _ = synthetic_experiment
    test_result = report["style_distance"]["mannwhitney"]

    assert test_result["alternative"] == "less"
    assert test_result["p_value"] < report["thresholds"]["mannwhitney_p_max"]


def test_drift_report_written(
    synthetic_experiment: tuple[dict[str, Any], Path],
) -> None:
    report, output_dir = synthetic_experiment
    drift = json.loads((output_dir / "drift.json").read_text(encoding="utf-8"))

    assert drift["schema_version"] == "1.0"
    assert {item["source_file"] for item in drift["files"]} == {
        f"author_{index}.txt" for index in range(5)
    }
    assert all(item["mean_distance"] >= 0 for item in drift["files"])
    assert set(report["split"]["author"]["all"]) == {
        f"author_{index}.txt" for index in range(5)
    }


def test_masked_file_excluded_from_training(
    synthetic_experiment: tuple[dict[str, Any], Path],
) -> None:
    report, output_dir = synthetic_experiment
    style = json.loads((output_dir / "style.json").read_text(encoding="utf-8"))
    masked_files = {"author_1.txt", "author_2.txt"}

    assert masked_files.isdisjoint(style["training_files"])
    assert masked_files.isdisjoint(report["discriminator"]["train_files"])
    assert masked_files <= set(style["excluded_files"])
    assert report["passport"]["n_chunks"] > 0


def test_unreviewed_masked_excerpt_fails(
    synthetic_experiment: tuple[dict[str, Any], Path],
) -> None:
    report, _ = synthetic_experiment
    golden = report["golden"]

    assert golden["reviewed_ids"] == ["reviewed"]
    assert golden["excluded_unreviewed_ids"] == ["not-reviewed"]
    assert [metric["id"] for metric in golden["metrics"]] == ["reviewed"]


def test_threshold_change_requires_version(tmp_path: Path) -> None:
    from stylometry.experiment import Thresholds, validate_threshold_change

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "report_version": "1.0.0",
                "thresholds": {"auc_min": 0.9, "mannwhitney_p_max": 0.01},
            }
        ),
        encoding="utf-8",
    )
    changed = Thresholds(auc_min=0.8, mannwhitney_p_max=0.01)

    with pytest.raises(ValueError, match="version"):
        validate_threshold_change(changed, "1.0.0", baseline, tuning_split="train")
    with pytest.raises(ValueError, match="held_out"):
        validate_threshold_change(changed, "2.0.0", baseline, tuning_split="held_out")
    with pytest.raises(ValueError, match="held_out"):
        validate_threshold_change(
            Thresholds(), "1.0.0", baseline, tuning_split="held_out"
        )
    with pytest.raises(ValueError, match="baseline|version"):
        validate_threshold_change(changed, "1.0.0", None, tuning_split="train")
    validate_threshold_change(changed, "2.0.0", baseline, tuning_split="train")


def test_malformed_review_metadata_fails() -> None:
    from stylometry.golden import GoldenExcerpt

    invalid_reviews = [
        {"reviewer": "human", "status": "approved", "reviewed_at": "not-a-date"},
        {
            "reviewer": "human",
            "status": "approved",
            "reviewed_at": "2026-11-27 06:00:00",
        },
        {"reviewer": "", "status": "approved", "reviewed_at": "2026-11-27T06:00:00Z"},
    ]
    for review in invalid_reviews:
        excerpt = GoldenExcerpt("id", "source.txt", "original", "masked", review)
        assert not excerpt.reviewed
