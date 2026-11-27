"""Deterministic Russian prose features with finite values for empty input."""

import math
import re
import statistics
from collections import Counter

from razdel import sentenize, tokenize  # type: ignore[import-untyped]

FEATURE_NAMES = (
    "avg_word_length",
    "word_length_std",
    "avg_sentence_words",
    "sentence_words_std",
    "lexical_diversity",
    "hapax_ratio",
    "pronoun_ratio",
    "conjunction_ratio",
    "preposition_ratio",
    "particle_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "punctuation_per_100_words",
    "commas_per_100_words",
    "semicolons_per_100_words",
    "colons_per_100_words",
    "questions_per_100_sentences",
    "exclamations_per_100_sentences",
    "ellipses_per_100_sentences",
    "dashes_per_100_sentences",
    "paragraphs_per_100_words",
    "repeated_word_ratio",
)

PRONOUNS = frozenset(
    [
        "я",
        "ты",
        "он",
        "она",
        "оно",
        "мы",
        "вы",
        "они",
        "меня",
        "тебя",
        "его",
        "ее",
        "нас",
        "вас",
        "их",
        "мне",
        "тебе",
        "ему",
        "ей",
        "нам",
        "вам",
        "им",
        "мной",
        "тобой",
        "им",
        "ею",
        "нами",
        "вами",
        "ими",
        "мой",
        "твой",
        "его",
        "наш",
        "ваш",
        "их",
        "свой",
        "себя",
        "себе",
        "собой",
    ]
)
CONJUNCTIONS = frozenset(
    [
        "и",
        "а",
        "но",
        "да",
        "или",
        "либо",
        "то",
        "что",
        "как",
        "когда",
        "если",
        "чтобы",
        "потому",
        "поэтому",
        "однако",
        "также",
        "же",
    ]
)
PREPOSITIONS = frozenset(
    [
        "в",
        "во",
        "на",
        "за",
        "под",
        "над",
        "у",
        "о",
        "об",
        "от",
        "до",
        "из",
        "из-за",
        "из-под",
        "с",
        "со",
        "по",
        "при",
        "про",
        "через",
        "между",
        "для",
        "без",
        "перед",
        "после",
        "около",
        "среди",
        "вместо",
        "вне",
        "внутри",
        "возле",
    ]
)
PARTICLES = frozenset(
    [
        "не",
        "нет",
        "ни",
        "же",
        "ж",
        "бы",
        "ли",
        "лишь",
        "только",
        "даже",
        "ведь",
        "вот",
        "именно",
        "разве",
    ]
)
ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)")
WORD_RE = re.compile(r"\w", re.UNICODE)


def _words(text: str) -> list[str]:
    return [
        token.text.lower().replace("ё", "е")
        for token in tokenize(text)
        if any(character.isalpha() for character in token.text)
    ]


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def extract_features(text: str) -> dict[str, float]:
    """Extract all FEATURE_NAMES in declaration order."""

    words = _words(text)
    word_count = len(words)
    sentences = list(sentenize(text))
    sentence_count = len(sentences)
    sentence_lengths = [len(_words(sentence.text)) for sentence in sentences]
    lengths = [len(word) for word in words]
    frequencies = Counter(words)
    letters = [character for character in text if character.isalpha()]
    punctuation_count = sum(character in ",;:!?—–-…().«»\"'" for character in text)

    features = {
        "avg_word_length": statistics.fmean(lengths) if lengths else 0.0,
        "word_length_std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "avg_sentence_words": (
            statistics.fmean(sentence_lengths) if sentence_lengths else 0.0
        ),
        "sentence_words_std": (
            statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
        ),
        "lexical_diversity": _ratio(len(frequencies), word_count),
        "hapax_ratio": _ratio(
            sum(count == 1 for count in frequencies.values()), word_count
        ),
        "pronoun_ratio": _ratio(sum(word in PRONOUNS for word in words), word_count),
        "conjunction_ratio": _ratio(
            sum(word in CONJUNCTIONS for word in words), word_count
        ),
        "preposition_ratio": _ratio(
            sum(word in PREPOSITIONS for word in words), word_count
        ),
        "particle_ratio": _ratio(sum(word in PARTICLES for word in words), word_count),
        "uppercase_ratio": _ratio(
            sum(character.isupper() for character in letters), len(letters)
        ),
        "digit_ratio": _ratio(
            sum(character.isdigit() for character in text), len(text)
        ),
        "punctuation_per_100_words": 100 * _ratio(punctuation_count, word_count),
        "commas_per_100_words": 100 * _ratio(text.count(","), word_count),
        "semicolons_per_100_words": 100 * _ratio(text.count(";"), word_count),
        "colons_per_100_words": 100 * _ratio(text.count(":"), word_count),
        "questions_per_100_sentences": 100 * _ratio(text.count("?"), sentence_count),
        "exclamations_per_100_sentences": 100 * _ratio(text.count("!"), sentence_count),
        "ellipses_per_100_sentences": 100
        * _ratio(len(ELLIPSIS_RE.findall(text)), sentence_count),
        "dashes_per_100_sentences": 100
        * _ratio(text.count("—") + text.count("–"), sentence_count),
        "paragraphs_per_100_words": 100
        * _ratio(len(re.findall(r"(?:\r?\n){2,}", text)), word_count),
        "repeated_word_ratio": _ratio(
            sum(count - 1 for count in frequencies.values()), word_count
        ),
    }
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("Feature declaration and extraction order diverged")
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Style features must be finite")
    return features
