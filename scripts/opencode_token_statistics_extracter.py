"""Extract aggregated token statistics from the OpenCode V2 SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

DEFAULT_DATABASE = Path.home() / ".local/share/opencode/opencode.db"


class ExtractorError(Exception):
    """Raised when the database cannot be read or contains malformed data."""


@dataclass(frozen=True)
class TimeBounds:
    start_ms: int | None
    end_ms: int | None


@dataclass
class TokenStatistics:
    sessions: set[str]
    messages: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning
            + self.cache_read
            + self.cache_write
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sum OpenCode V2 message token statistics.",
    )
    parser.add_argument("--session", help="Exact OpenCode session ID")
    parser.add_argument("--date", help="UTC calendar day in YYYY-MM-DD format")
    parser.add_argument("--from", dest="from_time", help="Inclusive ISO-8601 UTC bound")
    parser.add_argument("--to", dest="to_time", help="Exclusive ISO-8601 UTC bound")
    parser.add_argument("--project", help="Exact session_v2.directory value")
    parser.add_argument("--agent", help="Exact message data.agent value")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DATABASE),
        help=f"OpenCode database path (default: {DEFAULT_DATABASE})",
    )
    return parser


def parse_datetime(
    value: str,
    argument: str,
    parser: argparse.ArgumentParser,
) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parser.error(f"{argument} must be a valid ISO-8601 datetime")
    if parsed.tzinfo is None:
        parser.error(f"{argument} requires a timezone")
    return parsed.astimezone(UTC)


def validate_time_bounds(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> TimeBounds:
    if args.date is not None and (args.from_time is not None or args.to_time is not None):
        parser.error("--date cannot be combined with --from or --to")
    if args.session is None and args.date is None and args.from_time is None:
        parser.error("at least one selector is required: --session, --date or --from")
    if args.session is None and args.from_time is None and args.to_time is not None:
        parser.error("--to requires --session or --from")

    if args.date is not None:
        try:
            day = date.fromisoformat(args.date)
        except ValueError:
            parser.error("--date must use YYYY-MM-DD")
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        day_end = day_start + timedelta(days=1)
        return TimeBounds(int(day_start.timestamp() * 1000), int(day_end.timestamp() * 1000))
    else:
        start: datetime | None = None
        end: datetime | None = None

    if args.from_time is not None:
        start = parse_datetime(args.from_time, "--from", parser)
    if args.to_time is not None:
        end = parse_datetime(args.to_time, "--to", parser)
    if start is not None and end is not None and start >= end:
        parser.error("--from must be earlier than --to")

    return TimeBounds(
        int(start.timestamp() * 1000) if start is not None else None,
        int(end.timestamp() * 1000) if end is not None else None,
    )


def open_database(path: Path) -> sqlite3.Connection:
    database_path = path.expanduser()
    if not database_path.is_file():
        raise ExtractorError(f"database does not exist: {database_path}")
    try:
        connection = sqlite3.connect(
            f"file:{quote(str(database_path.resolve()), safe='/')}?mode=ro",
            uri=True,
        )
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.Error as exc:
        raise ExtractorError(f"cannot open database read-only: {exc}") from exc
    return connection


def token_metric(
    container: dict[str, Any],
    key: str,
    message_id: str,
) -> int:
    value = container.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExtractorError(
            f"message {message_id}: tokens.{key} must be an integer",
        )
    return value


def add_message_tokens(
    statistics: TokenStatistics,
    message_id: str,
    session_id: str,
    tokens: Any,
) -> None:
    if not isinstance(tokens, dict):
        raise ExtractorError(f"message {message_id}: tokens must be an object")
    cache = tokens.get("cache") or {}
    if not isinstance(cache, dict):
        raise ExtractorError(f"message {message_id}: tokens.cache must be an object")

    statistics.sessions.add(session_id)
    statistics.messages += 1
    statistics.input_tokens += token_metric(tokens, "input", message_id)
    statistics.output_tokens += token_metric(tokens, "output", message_id)
    statistics.reasoning += token_metric(tokens, "reasoning", message_id)
    statistics.cache_read += token_metric(cache, "read", message_id)
    statistics.cache_write += token_metric(cache, "write", message_id)


def extract_statistics(
    database_path: Path,
    *,
    session: str | None,
    project: str | None,
    agent: str | None,
    bounds: TimeBounds,
) -> TokenStatistics:
    conditions = ["m.type = ?"]
    parameters: list[object] = ["assistant"]
    if session is not None:
        conditions.append("m.session_id = ?")
        parameters.append(session)
    if project is not None:
        conditions.append("s.directory = ?")
        parameters.append(project)
    if bounds.start_ms is not None:
        conditions.append("m.time_created >= ?")
        parameters.append(bounds.start_ms)
    if bounds.end_ms is not None:
        conditions.append("m.time_created < ?")
        parameters.append(bounds.end_ms)

    query = f"""
        SELECT m.id, m.session_id, m.data
        FROM session_message AS m
        JOIN session_v2 AS s ON s.id = m.session_id
        WHERE {' AND '.join(conditions)}
        ORDER BY m.time_created, m.id
    """

    connection = open_database(database_path)
    try:
        try:
            rows = connection.execute(query, parameters).fetchall()
        except sqlite3.Error as exc:
            raise ExtractorError(f"cannot read OpenCode V2 messages: {exc}") from exc
    finally:
        connection.close()

    statistics = TokenStatistics(sessions=set())
    for message_id, session_id, raw_data in rows:
        try:
            data = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ExtractorError(f"message {message_id}: invalid JSON data") from exc
        if not isinstance(data, dict):
            raise ExtractorError(f"message {message_id}: data must be an object")
        if agent is not None and data.get("agent") != agent:
            continue
        tokens = data.get("tokens")
        if tokens is None:
            continue
        add_message_tokens(statistics, message_id, session_id, tokens)
    return statistics


def print_statistics(statistics: TokenStatistics) -> None:
    print(f"Sessions: {len(statistics.sessions)}")
    print(f"Messages: {statistics.messages}")
    print(f"Токены in: {statistics.input_tokens}")
    print(f"Токены out: {statistics.output_tokens}")
    print(f"Reasoning: {statistics.reasoning}")
    print(f"Cache read: {statistics.cache_read}")
    print(f"Cache write: {statistics.cache_write}")
    print(f"Токены всего: {statistics.total}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    bounds = validate_time_bounds(args, parser)
    try:
        statistics = extract_statistics(
            Path(args.db),
            session=args.session,
            project=args.project,
            agent=args.agent,
            bounds=bounds,
        )
    except ExtractorError as exc:
        print(f"opencode-token-statistics: error: {exc}", file=sys.stderr)
        return 1
    print_statistics(statistics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
