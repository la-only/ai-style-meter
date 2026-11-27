"""Offline integrity checks for the committed public-domain baseline."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "stylometry"


def test_golden_copies_match_read_only_source() -> None:
    golden_path = DATA / "golden" / "masked.jsonl"
    entries = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries
    for entry in entries:
        assert entry["schema_version"] == "1.0"
        source = (DATA / "corpus" / "chekhov" / entry["source_file"]).read_text(
            encoding="utf-8"
        )
        masked_copy = (
            (DATA / "golden" / entry["masked_file"]).read_text(encoding="utf-8").strip()
        )
        assert entry["original"] in source
        assert masked_copy == entry["masked"]
        assert "[MASK]" in entry["masked"]
        assert "review" not in entry, "The agent must not fabricate human review"


def test_corpus_has_no_duplicated_long_narrative_paragraphs() -> None:
    for path in (DATA / "corpus").glob("*/*.txt"):
        paragraphs = [
            paragraph.strip()
            for paragraph in path.read_text(encoding="utf-8").split("\n\n")
            if sum(
                any(character.isalpha() for character in word)
                for word in paragraph.split()
            )
            >= 20
        ]
        assert len(paragraphs) == len(set(paragraphs)), (
            f"Duplicated source content: {path}"
        )


def test_committed_baseline_is_leakage_free_and_meets_frozen_thresholds() -> None:
    report = json.loads(
        (DATA / "baseline" / "experiment.json").read_text(encoding="utf-8")
    )
    style = json.loads((DATA / "baseline" / "style.json").read_text(encoding="utf-8"))
    drift = json.loads((DATA / "baseline" / "drift.json").read_text(encoding="utf-8"))
    config = json.loads((DATA / "baseline_config.json").read_text(encoding="utf-8"))
    golden_entries = [
        json.loads(line)
        for line in (DATA / "golden" / "masked.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    quarantined = {entry["source_file"] for entry in golden_entries}

    assert report["report_version"] == config["report_version"] == "1.0.0"
    assert report["thresholds"] == config["thresholds"]
    assert report["tuning_split"] == "train"
    assert report["passed"] is True
    assert report["discriminator"]["auc"] >= config["thresholds"]["auc_min"]
    assert (
        report["style_distance"]["mannwhitney"]["p_value"]
        < config["thresholds"]["mannwhitney_p_max"]
    )
    author_split = report["split"]["author"]
    assert set(author_split["train"]).isdisjoint(author_split["held_out"])
    assert set(report["discriminator"]["train_files"]).isdisjoint(
        report["discriminator"]["held_out_files"]
    )
    assert quarantined.isdisjoint(style["training_files"])
    assert quarantined.isdisjoint(report["discriminator"]["train_files"])
    assert style["n_chunks"] >= 100
    assert all(
        math.isfinite(stats["mean"])
        and math.isfinite(stats["std"])
        and stats["std"] > 0
        for stats in style["features"].values()
    )
    assert {item["source_file"] for item in drift["files"]} == set(author_split["all"])
    assert report["golden"]["metrics"] == []
    assert set(report["golden"]["excluded_unreviewed_ids"]) == {
        entry["id"] for entry in golden_entries
    }
