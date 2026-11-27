"""Command-line interface for cards, scoring, and frozen experiments."""

import json
from pathlib import Path

import click

from stylometry.experiment import ExperimentConfig, Thresholds, run_experiment
from stylometry.golden import load_golden
from stylometry.models import chunk_documents, load_documents
from stylometry.passport import (
    build_style_passport,
    distance_to_passport,
    load_style_passport,
    save_style_passport,
)


@click.group()
def main() -> None:
    """Measure authorial style without a generative rewriting loop."""


@main.command("card")
@click.option(
    "--corpus",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--author", required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--golden", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--chunk-words", default=200, show_default=True, type=click.IntRange(min=1)
)
def card_command(
    corpus: Path,
    author: str,
    output: Path,
    golden: Path | None,
    chunk_words: int,
) -> None:
    """Build a style.json from all non-quarantined corpus files."""

    excluded = {excerpt.source_file for excerpt in load_golden(golden)}
    documents = [
        document
        for document in load_documents(corpus)
        if document.source_file not in excluded
    ]
    chunks = chunk_documents(documents, chunk_words=chunk_words)
    passport = build_style_passport(
        chunks,
        author=author,
        chunk_words=chunk_words,
        excluded_files=tuple(sorted(excluded)),
    )
    save_style_passport(passport, output)
    click.echo(json.dumps({"style": str(output), "n_chunks": passport.n_chunks}))


@main.command("score")
@click.option(
    "--passport",
    "passport_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def score_command(passport_path: Path, input_path: Path) -> None:
    """Print the mean absolute z-distance of a UTF-8 text to a passport."""

    passport = load_style_passport(passport_path)
    distance = distance_to_passport(input_path.read_text(encoding="utf-8"), passport)
    click.echo(json.dumps({"distance": distance, "author": passport.author}))


@main.command("experiment")
@click.option(
    "--corpus",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--other",
    "other_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    required=True,
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--golden", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--baseline-report", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--report-version", default="1.0.0", show_default=True)
@click.option("--auc-min", default=0.90, show_default=True, type=float)
@click.option("--mannwhitney-p-max", default=0.01, show_default=True, type=float)
@click.option(
    "--chunk-words", default=200, show_default=True, type=click.IntRange(min=1)
)
def experiment_command(
    corpus: Path,
    other_dirs: tuple[Path, ...],
    output: Path,
    golden: Path | None,
    baseline_report: Path | None,
    report_version: str,
    auc_min: float,
    mannwhitney_p_max: float,
    chunk_words: int,
) -> None:
    """Run a whole-file held-out experiment and write three JSON artifacts."""

    report = run_experiment(
        author_dir=corpus,
        other_dirs=list(other_dirs),
        output_dir=output,
        golden_path=golden,
        baseline_report=baseline_report,
        config=ExperimentConfig(
            thresholds=Thresholds(auc_min=auc_min, mannwhitney_p_max=mannwhitney_p_max),
            report_version=report_version,
            chunk_words=chunk_words,
        ),
    )
    click.echo(
        json.dumps(
            {
                "experiment": str(output / "experiment.json"),
                "auc": report["discriminator"]["auc"],
                "p_value": report["style_distance"]["mannwhitney"]["p_value"],
                "passed": report["passed"],
            }
        )
    )
    if not report["passed"]:
        raise click.exceptions.Exit(1)


if __name__ == "__main__":
    main()
