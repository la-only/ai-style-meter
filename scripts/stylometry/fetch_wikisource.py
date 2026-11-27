"""Fetch the declared public-domain Wikisource corpus into data/stylometry.

This acquisition utility is never called by tests or by the experiment itself.
"""

import http.client
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from razdel import tokenize  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data" / "stylometry" / "corpus"
USER_AGENT = "ai-style-meter/0.1 (public-domain stylometry research)"
SOURCES = {
    "chekhov": {
        "01_skuchnaya_istoriya.txt": "Скучная история (Чехов)",
        "02_step.txt": "Степь (Чехов)",
        "03_palata_6.txt": "Палата № 6 (Чехов)",
        "04_v_ovrage.txt": "В овраге (Чехов)",
        "05_duel.txt": "Дуэль (Чехов)",
        "06_tri_goda.txt": "Три года (Чехов)",
    },
    "dostoevsky": {
        "01_bednye_lyudi.txt": "Бедные люди (Достоевский)",
        "02_dvoynik.txt": "Двойник (Достоевский)",
        "03_belye_nochi.txt": "Белые ночи (Достоевский)",
        "04_zapiski_iz_podpolya.txt": "Записки из подполья (Достоевский)",
        "05_igrok.txt": "Игрок (Достоевский)",
    },
    "technical": {
        "01_arifmetika.txt": "ЭСБЕ/Арифметика",
        "02_algebra.txt": "ЭСБЕ/Алгебра",
        "03_geometriya.txt": "ЭСБЕ/Геометрия",
        "04_mechanika.txt": "ЭСБЕ/Механика",
        "05_fizika.txt": "ЭСБЕ/Физика",
        "06_himiya.txt": "ЭСБЕ/Химия",
    },
}


def _api(parameters: dict[str, str]) -> dict[str, Any]:
    query = "/w/api.php?" + urllib.parse.urlencode(parameters)
    for attempt in range(5):
        time.sleep(1.0)
        connection = http.client.HTTPSConnection("ru.wikisource.org", timeout=60)
        connection.request("GET", query, headers={"User-Agent": USER_AGENT})
        response = connection.getresponse()
        status = response.status
        payload = response.read()
        connection.close()
        if status == 200:
            return json.loads(payload)  # type: ignore[no-any-return]
        if status != 429 or attempt == 4:
            raise RuntimeError(f"Wikisource HTTP {status}: {payload[:200]!r}")
        time.sleep(5.0 * (attempt + 1))
    raise RuntimeError("Unreachable retry state")


def _word_count(text: str) -> int:
    return sum(
        any(character.isalpha() for character in token.text) for token in tokenize(text)
    )


def _extract_text(title: str, *, allow_subpages: bool = True) -> str:
    payload = _api(
        {
            "action": "parse",
            "page": title,
            "prop": "text|links",
            "format": "json",
            "formatversion": "2",
        }
    )
    if "error" in payload:
        raise RuntimeError(f"Wikisource rejected {title}: {payload['error']}")
    soup = BeautifulSoup(payload["parse"]["text"], "html.parser")
    for element in soup.select(
        "table, sup, style, script, .mw-editsection, .noprint, .ws-noexport, .references"
    ):
        element.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    clean_lines = [
        line
        for line in lines
        if line and not re.fullmatch(r"(?:\d+|[IVXLCDM]+)", line, flags=re.IGNORECASE)
    ]
    clean_text = "\n\n".join(clean_lines) + "\n"
    if _word_count(clean_text) < 500 and allow_subpages:
        prefix = title + "/"
        subpages: list[str] = []
        for link in payload["parse"].get("links", []):
            linked_title = str(link.get("title", link.get("*", "")))
            if (
                link.get("ns") == 0
                and linked_title.startswith(prefix)
                and linked_title not in subpages
            ):
                subpages.append(linked_title)
        full_text_pages = [
            subpage for subpage in subpages if subpage == prefix + "Весь текст"
        ]
        if full_text_pages:
            subpages = full_text_pages
        if subpages:
            chapters = [
                _extract_text(subpage, allow_subpages=False) for subpage in subpages
            ]
            clean_text = "\n\n".join(chapters)
    if _word_count(clean_text) < 100:
        raise RuntimeError(
            f"Extracted text is unexpectedly short ({_word_count(clean_text)} words): {title}"
        )
    return clean_text


def main() -> None:
    """Download each source once and emit an attribution manifest."""

    manifest: list[dict[str, Any]] = []
    for corpus, sources in SOURCES.items():
        output_dir = OUTPUT_ROOT / corpus
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, title in sources.items():
            text = _extract_text(title)
            (output_dir / filename).write_text(text, encoding="utf-8")
            manifest.append(
                {
                    "corpus": corpus,
                    "file": filename,
                    "source_page": title,
                    "source_url": "https://ru.wikisource.org/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_"), safe="()/№"),
                    "rights": "Public-domain original; Chekhov/Dostoevsky or pre-1917 Brockhaus and Efron",
                    "encoding": "UTF-8",
                }
            )
            print(f"{corpus}/{filename}: {len(text)} characters")
    manifest_path = OUTPUT_ROOT.parent / "sources.json"
    manifest_path.write_text(
        json.dumps(
            {"schema_version": "1.0", "sources": manifest},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
