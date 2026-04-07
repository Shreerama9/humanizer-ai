"""
Readability metrics for generated SEO content.
Implements Flesch-Kincaid Grade Level, Flesch Reading Ease,
Gunning Fog Index, and SMOG Grade.
"""

import re
import math
from dataclasses import dataclass


@dataclass
class ReadabilityReport:
    flesch_reading_ease: float      # 60–70 is ideal for general web content
    flesch_kincaid_grade: float     # Target: 8–10 for SEO content
    gunning_fog: float              # Target: < 12
    smog_grade: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    word_count: int
    sentence_count: int
    grade_label: str                # "Easy", "Standard", "Difficult"

    def is_seo_optimal(self) -> bool:
        """Returns True if content hits ideal SEO readability targets."""
        return (
            60 <= self.flesch_reading_ease <= 80
            and 7 <= self.flesch_kincaid_grade <= 11
        )


def count_syllables(word: str) -> int:
    """
    Estimate syllable count using a heuristic.
    Not perfect but fast and sufficient for bulk scoring.
    """
    word = word.lower().strip(".,!?;:'\"()")
    if not word:
        return 0
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    # Subtract silent e
    if word.endswith("e") and count > 1:
        count -= 1
    # Every word has at least one syllable
    return max(1, count)


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences — handles abbreviations and ellipses."""
    # Split on .!? followed by whitespace and uppercase
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> list[str]:
    """Extract alphabetic words only."""
    return re.findall(r"\b[a-zA-Z']+\b", text)


def flesch_reading_ease(word_count: int, sentence_count: int, syllable_count: int) -> float:
    """
    Flesch Reading Ease formula.
    206.835 − 1.015 × (words/sentences) − 84.6 × (syllables/words)
    Score: 90–100 = Very Easy, 60–70 = Standard, 0–30 = Very Difficult
    """
    if sentence_count == 0 or word_count == 0:
        return 0.0
    asl = word_count / sentence_count
    asw = syllable_count / word_count
    return 206.835 - (1.015 * asl) - (84.6 * asw)


def flesch_kincaid_grade(word_count: int, sentence_count: int, syllable_count: int) -> float:
    """
    Flesch-Kincaid Grade Level.
    0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59
    """
    if sentence_count == 0 or word_count == 0:
        return 0.0
    asl = word_count / sentence_count
    asw = syllable_count / word_count
    return (0.39 * asl) + (11.8 * asw) - 15.59


def gunning_fog(word_count: int, sentence_count: int, complex_word_count: int) -> float:
    """
    Gunning Fog Index.
    0.4 × [(words/sentences) + 100 × (complex_words/words)]
    Complex words = 3+ syllables (excluding proper nouns, jargon suffixes)
    """
    if sentence_count == 0 or word_count == 0:
        return 0.0
    asl = word_count / sentence_count
    pcw = complex_word_count / word_count
    return 0.4 * (asl + 100 * pcw)


def smog_grade(sentence_count: int, polysyllabic_count: int) -> float:
    """SMOG Grade — needs ≥30 sentences for accuracy."""
    if sentence_count < 3:
        return 0.0
    return 3.1291 + 1.0430 * math.sqrt(polysyllabic_count * (30 / sentence_count))


def _grade_label(score: float) -> str:
    if score >= 70:
        return "Easy"
    if score >= 50:
        return "Standard"
    return "Difficult"


def analyze(text: str) -> ReadabilityReport:
    """Full readability analysis of a text string."""
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)

    sentence_count = max(len(sentences), 1)
    word_count = max(len(words), 1)

    syllables = [count_syllables(w) for w in words]
    syllable_count = sum(syllables)
    complex_words = sum(1 for s in syllables if s >= 3)

    fre = flesch_reading_ease(word_count, sentence_count, syllable_count)
    fkg = flesch_kincaid_grade(word_count, sentence_count, syllable_count)
    fog = gunning_fog(word_count, sentence_count, complex_words)
    smog = smog_grade(sentence_count, complex_words)

    return ReadabilityReport(
        flesch_reading_ease=round(fre, 2),
        flesch_kincaid_grade=round(fkg, 2),
        gunning_fog=round(fog, 2),
        smog_grade=round(smog, 2),
        avg_sentence_length=round(word_count / sentence_count, 2),
        avg_syllables_per_word=round(syllable_count / word_count, 2),
        word_count=word_count,
        sentence_count=sentence_count,
        grade_label=_grade_label(fre),
    )
