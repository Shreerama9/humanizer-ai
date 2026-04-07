"""
SEO Content Evaluator

Scores generated content across key SEO signals:
  - Keyword density (primary + secondary)
  - Entity coverage and Named Entity Recognition density
  - Header structure (H1/H2/H3 hierarchy)
  - Meta description presence and length
  - Internal linking potential (anchor-worthy phrases)
  - Content structure score (intro, body, conclusion pattern)
  - E-E-A-T signals (Experience, Expertise, Authoritativeness, Trustworthiness)
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


@dataclass
class SEOReport:
    overall_seo_score: float         # 0–100
    keyword_density: float           # Primary keyword density %
    keyword_prominence: float        # Is keyword in H1/first 100 words?
    secondary_keyword_coverage: float  # Fraction of secondary kws present
    header_structure_score: float    # Proper H2/H3 hierarchy
    meta_description_present: bool
    meta_description_length: int     # Ideal: 150–160 chars
    word_count: int
    entity_density: float            # Named entities per 100 words
    eeat_signals: list[str]          # Detected E-E-A-T signals
    content_structure_score: float   # Intro/body/conclusion detection
    issues: list[str] = field(default_factory=list)

    def is_seo_optimized(self) -> bool:
        return self.overall_seo_score >= 70.0

    def summary(self) -> str:
        status = "OPTIMIZED" if self.is_seo_optimized() else "NEEDS WORK"
        return (
            f"[{status}] SEO Score: {self.overall_seo_score:.1f}/100 | "
            f"KW Density: {self.keyword_density:.2f}% | "
            f"Words: {self.word_count} | "
            f"Entities/100w: {self.entity_density:.1f}"
        )


class SEOEvaluator:
    """Evaluates SEO quality of generated content without external APIs."""

    EEAT_PATTERNS = {
        "experience": [
            r"\bI (have|worked|built|used|tried|tested|spent)\b",
            r"\bin my experience\b",
            r"\bfrom (my|our) (experience|perspective|work)\b",
            r"\bI'(ve|d) (seen|found|noticed)\b",
        ],
        "expertise": [
            r"\baccording to (research|studies|data)\b",
            r"\b(studies|research|data) (show|suggest|indicate|found)\b",
            r"\b\d{4} (study|report|survey|research)\b",
            r"\bexperts? (say|recommend|suggest|agree)\b",
        ],
        "authoritativeness": [
            r"\b(published|featured|cited) in\b",
            r"\bPh\.?D|M\.?D|MBA|CEO|CTO\b",
            r"\b\d+ years? of (experience|expertise)\b",
        ],
        "trustworthiness": [
            r"\bsources?:\s",
            r"\breferences?:\s",
            r"\bhttps?://\b",
            r"\b\[\d+\]",  # Citation markers
            r"\bdisclosure:\b",
        ],
    }

    def __init__(self):
        self._nlp = None
        if _SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                pass  # Model not downloaded — graceful degradation

    # ── Keyword analysis ──────────────────────────────────────────────────────

    def _keyword_density(self, text: str, keyword: str) -> float:
        words = re.findall(r"\b\w+\b", text.lower())
        kw_words = keyword.lower().split()
        count = sum(
            1 for i in range(len(words) - len(kw_words) + 1)
            if words[i:i+len(kw_words)] == kw_words
        )
        return (count / max(len(words), 1)) * 100

    def _keyword_prominence(self, text: str, keyword: str) -> float:
        """Score based on keyword appearing in headers and early text."""
        score = 0.0
        kw = keyword.lower()
        lines = text.lower().split("\n")

        # Check H1/H2 (first ## heading)
        for line in lines:
            if re.match(r"^#{1,2}\s", line) and kw in line:
                score += 40
                break

        # Check first 100 words
        first_100 = " ".join(text.split()[:100]).lower()
        if kw in first_100:
            score += 30

        # Check title/meta description comment
        meta_match = re.search(r"<!--\s*meta:\s*(.*?)\s*-->", text, re.IGNORECASE)
        if meta_match and kw in meta_match.group(1).lower():
            score += 20

        # Check URL-like slug patterns
        slug = kw.replace(" ", "-")
        if slug in text.lower():
            score += 10

        return min(100.0, score)

    def _secondary_keyword_coverage(self, text: str, secondary_keywords: list[str]) -> float:
        if not secondary_keywords:
            return 1.0
        text_lower = text.lower()
        found = sum(1 for kw in secondary_keywords if kw.lower() in text_lower)
        return found / len(secondary_keywords)

    # ── Structure analysis ────────────────────────────────────────────────────

    def _header_structure_score(self, text: str) -> float:
        h2_count = len(re.findall(r"^##\s", text, re.MULTILINE))
        h3_count = len(re.findall(r"^###\s", text, re.MULTILINE))
        h1_count = len(re.findall(r"^#\s", text, re.MULTILINE))
        score = 0.0

        if h2_count >= 2:
            score += 50
        if h3_count >= 1:
            score += 25
        if h1_count <= 1:  # Only one H1 is good SEO practice
            score += 25
        if h2_count > 8:   # Too many H2s dilute focus
            score -= 10

        return min(100.0, max(0.0, score))

    def _meta_description(self, text: str) -> tuple[bool, int]:
        match = re.search(r"<!--\s*meta:\s*(.*?)\s*-->", text, re.IGNORECASE)
        if match:
            return True, len(match.group(1))
        return False, 0

    def _content_structure_score(self, text: str) -> float:
        """Check for intro paragraph, body sections, and conclusion."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        score = 0.0

        if len(paragraphs) >= 3:
            score += 30  # Has multiple paragraphs
        if len(paragraphs) >= 5:
            score += 20  # Well-structured

        # Check for conclusion signals
        conclusion_patterns = [
            r"\b(in conclusion|to summarize|final thoughts|wrapping up|bottom line)\b",
            r"\b(takeaway|key (takeaways|points|lessons))\b",
        ]
        text_lower = text.lower()
        for pat in conclusion_patterns:
            if re.search(pat, text_lower):
                score += 25
                break

        # Check for intro hook (question or bold statement in first para)
        if paragraphs:
            first_para = paragraphs[0]
            if "?" in first_para or re.search(r"\*\*.+\*\*", first_para):
                score += 25

        return min(100.0, score)

    # ── E-E-A-T signals ───────────────────────────────────────────────────────

    def _detect_eeat(self, text: str) -> list[str]:
        signals = []
        for category, patterns in self.EEAT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    signals.append(category)
                    break
        return signals

    # ── Entity density ────────────────────────────────────────────────────────

    def _entity_density(self, text: str) -> float:
        """Named entities per 100 words."""
        if self._nlp:
            doc = self._nlp(text[:5000])  # Limit for speed
            entity_count = len(doc.ents)
        else:
            # Fallback: count capitalized multi-word phrases
            entity_count = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text))

        word_count = max(len(text.split()), 1)
        return (entity_count / word_count) * 100

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        text: str,
        primary_keyword: str,
        secondary_keywords: Optional[list[str]] = None,
    ) -> SEOReport:
        secondary_keywords = secondary_keywords or []
        word_count = len(re.findall(r"\b\w+\b", text))

        density = self._keyword_density(text, primary_keyword)
        prominence = self._keyword_prominence(text, primary_keyword)
        sec_coverage = self._secondary_keyword_coverage(text, secondary_keywords)
        header_score = self._header_structure_score(text)
        meta_present, meta_len = self._meta_description(text)
        content_score = self._content_structure_score(text)
        entity_dens = self._entity_density(text)
        eeat = self._detect_eeat(text)

        issues = []
        if density < 0.5:
            issues.append(f"Keyword density too low: {density:.2f}% (target: 1–2%)")
        if density > 3.0:
            issues.append(f"Keyword stuffing risk: {density:.2f}% (target: 1–2%)")
        if word_count < 500:
            issues.append(f"Content too short: {word_count} words (target: 800+)")
        if not meta_present:
            issues.append("Missing meta description comment")
        elif not (150 <= meta_len <= 165):
            issues.append(f"Meta description length {meta_len} (ideal: 150–160 chars)")
        if header_score < 50:
            issues.append("Insufficient header structure — add more H2/H3 sections")

        # Meta description score
        meta_score = 0.0
        if meta_present:
            meta_score = 80.0 if 150 <= meta_len <= 165 else 50.0

        # Weighted overall score
        overall = (
            min(100, (1.5 - abs(density - 1.5)) / 1.5 * 100) * 0.20  # Keyword density
            + prominence * 0.15
            + sec_coverage * 100 * 0.10
            + header_score * 0.15
            + meta_score * 0.10
            + content_score * 0.15
            + min(100, entity_dens * 10) * 0.10
            + min(100, len(eeat) * 25) * 0.05
        )

        return SEOReport(
            overall_seo_score=round(overall, 2),
            keyword_density=round(density, 3),
            keyword_prominence=round(prominence, 2),
            secondary_keyword_coverage=round(sec_coverage, 3),
            header_structure_score=round(header_score, 2),
            meta_description_present=meta_present,
            meta_description_length=meta_len,
            word_count=word_count,
            entity_density=round(entity_dens, 2),
            eeat_signals=eeat,
            content_structure_score=round(content_score, 2),
            issues=issues,
        )

    def evaluate_batch(
        self,
        texts: list[str],
        primary_keyword: str,
        secondary_keywords: Optional[list[str]] = None,
    ) -> list[SEOReport]:
        return [self.evaluate(t, primary_keyword, secondary_keywords) for t in texts]
