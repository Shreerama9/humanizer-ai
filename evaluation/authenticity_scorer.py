"""
AI Content Authenticity Scorer

Implements a multi-signal approach to detect and score AI-generated text
without relying on external black-box detectors. Uses:
  1. Perplexity (via loaded model) — AI text has low, uniform perplexity
  2. Burstiness — AI text has low burstiness (uniform sentence lengths)
  3. Vocabulary richness — AI text often reuses limited vocabulary
  4. Sentence-start diversity — AI text repeats sentence openers
  5. Punctuation patterns — AI overuses certain patterns (em-dash, semicolons)
  6. n-gram repetition — duplicate phrases signal AI templating

Combining these gives an interpretable "humanness score" 0–100.
Higher = more human-like.
"""

import re
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AuthenticityReport:
    overall_score: float              # 0–100; >75 considered "passes detection"
    perplexity_score: float           # normalized; higher = more human-like
    burstiness_score: float           # sentence length variance
    vocabulary_richness: float        # type-token ratio
    sentence_start_diversity: float   # unique first words / total sentences
    ngram_repetition_score: float     # inverse of repeated n-gram ratio
    punctuation_naturalness: float    # based on em-dash, semicolon, ellipsis usage
    flags: list[str]                  # human-readable warnings

    def passes_detection(self, threshold: float = 75.0) -> bool:
        return self.overall_score >= threshold

    def summary(self) -> str:
        status = "PASSES" if self.passes_detection() else "DETECTED AS AI"
        return (
            f"[{status}] Humanness Score: {self.overall_score:.1f}/100 | "
            f"PPL: {self.perplexity_score:.1f} | "
            f"Burstiness: {self.burstiness_score:.1f} | "
            f"Vocab TTR: {self.vocabulary_richness:.2f}"
        )


class AuthenticityScorer:
    """
    Scores content authenticity using linguistic signals.
    Optionally uses a local language model for perplexity computation.
    """

    # Thresholds derived from analysis of GPT-4/Llama outputs vs human writing
    AI_PERPLEXITY_THRESHOLD = 25.0   # AI text typically PPL < 20
    HUMAN_PERPLEXITY_TYPICAL = 60.0

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self._model = None
        self._tokenizer = None
        if model_id:
            self._load_ppl_model(model_id)

    def _load_ppl_model(self, model_id: str):
        """Load a small reference model for perplexity computation."""
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self._model.eval()

    @torch.no_grad()
    def compute_perplexity(self, text: str, stride: int = 512) -> float:
        """
        Sliding-window perplexity — handles texts longer than model context.
        Lower PPL = more predictable = more likely AI-generated.
        """
        if self._model is None:
            return -1.0  # Unavailable

        encodings = self._tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        max_len = self._model.config.max_position_embeddings
        seq_len = input_ids.size(1)

        nlls = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_len, seq_len)
            target_len = end - prev_end
            input_chunk = input_ids[:, begin:end]
            target_ids = input_chunk.clone()
            target_ids[:, :-target_len] = -100  # Mask context tokens

            outputs = self._model(input_chunk, labels=target_ids)
            nlls.append(outputs.loss * target_len)
            prev_end = end
            if end == seq_len:
                break

        ppl = math.exp(torch.stack(nlls).sum() / seq_len)
        return float(ppl)

    # ── Linguistic signals ────────────────────────────────────────────────────

    def _burstiness(self, text: str) -> float:
        """
        Burstiness = std(sentence_lengths) / mean(sentence_lengths).
        Human text is bursty (high variance); AI text is uniform (low variance).
        Normalized 0–100 where 100 = maximally bursty (human-like).
        """
        sentences = re.split(r"[.!?]+", text)
        lengths = [len(s.split()) for s in sentences if len(s.split()) > 1]
        if len(lengths) < 3:
            return 50.0
        mean = statistics.mean(lengths)
        if mean == 0:
            return 0.0
        cv = statistics.stdev(lengths) / mean  # Coefficient of variation
        # CV > 0.8 is very bursty (human); CV < 0.3 is uniform (AI)
        return min(100.0, cv * 100)

    def _vocabulary_richness(self, text: str) -> float:
        """
        Type-Token Ratio (TTR) over a 100-word sliding window (MSTTR).
        Higher = richer vocabulary = more human-like.
        Human TTR typically 0.65–0.85; AI tends toward 0.5–0.65.
        """
        words = re.findall(r"\b[a-z]+\b", text.lower())
        if len(words) < 10:
            return 0.5
        window = 100
        ttrs = []
        for i in range(0, len(words) - window, window):
            chunk = words[i:i + window]
            ttrs.append(len(set(chunk)) / len(chunk))
        return statistics.mean(ttrs) if ttrs else len(set(words)) / len(words)

    def _sentence_start_diversity(self, text: str) -> float:
        """Fraction of unique first words across sentences."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) < 3:
            return 1.0
        first_words = [s.split()[0].lower() for s in sentences if s.split()]
        return len(set(first_words)) / len(first_words)

    def _ngram_repetition(self, text: str, n: int = 5) -> float:
        """
        Returns 1 - (repeated_ngram_ratio).
        AI text often contains repeated phrases.
        Score of 1.0 = no repetition (more human-like).
        """
        words = re.findall(r"\b[a-z]+\b", text.lower())
        if len(words) < n * 2:
            return 1.0
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n)]
        unique = len(set(ngrams))
        return unique / len(ngrams)

    def _punctuation_naturalness(self, text: str) -> float:
        """
        AI models overuse em-dashes, semicolons, and colon-list patterns.
        Scores based on deviation from natural human usage rates.
        """
        word_count = max(len(text.split()), 1)
        em_dashes = text.count("—") + text.count(" - ")
        semicolons = text.count(";")
        ellipses = text.count("...")
        colons = text.count(":")

        # Rates per 1000 words
        em_rate = (em_dashes / word_count) * 1000
        semi_rate = (semicolons / word_count) * 1000
        colon_rate = (colons / word_count) * 1000

        # AI tends toward: em-dash > 5/1k, semicolons > 3/1k
        score = 100.0
        if em_rate > 8:
            score -= 20
        elif em_rate > 5:
            score -= 10
        if semi_rate > 5:
            score -= 15
        elif semi_rate > 3:
            score -= 7
        if colon_rate > 10:
            score -= 10
        return max(0.0, score)

    def _build_flags(self, report_data: dict) -> list[str]:
        flags = []
        if report_data["perplexity"] > 0 and report_data["perplexity"] < 20:
            flags.append("Very low perplexity — highly predictable text pattern")
        if report_data["burstiness"] < 30:
            flags.append("Low sentence-length variance — uniform AI-like rhythm")
        if report_data["ttr"] < 0.55:
            flags.append("Low vocabulary richness — repetitive word choice")
        if report_data["start_diversity"] < 0.6:
            flags.append("Low sentence-start diversity — many sentences begin the same way")
        if report_data["ngram"] < 0.85:
            flags.append("High n-gram repetition — templated phrases detected")
        return flags

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, text: str) -> AuthenticityReport:
        ppl = self.compute_perplexity(text)
        burstiness = self._burstiness(text)
        ttr = self._vocabulary_richness(text)
        start_div = self._sentence_start_diversity(text)
        ngram = self._ngram_repetition(text)
        punct = self._punctuation_naturalness(text)

        # Normalize perplexity to 0–100 score
        if ppl < 0:
            ppl_score = 50.0  # Unknown — model not loaded
        else:
            ppl_score = min(100.0, max(0.0, (ppl - 10) / (80 - 10) * 100))

        flags = self._build_flags({
            "perplexity": ppl, "burstiness": burstiness,
            "ttr": ttr, "start_diversity": start_div, "ngram": ngram,
        })

        # Weighted aggregate
        overall = (
            ppl_score * 0.30
            + burstiness * 0.25
            + ttr * 100 * 0.20
            + start_div * 100 * 0.10
            + ngram * 100 * 0.10
            + punct * 0.05
        )

        return AuthenticityReport(
            overall_score=round(min(100.0, overall), 2),
            perplexity_score=round(ppl_score, 2),
            burstiness_score=round(burstiness, 2),
            vocabulary_richness=round(ttr, 4),
            sentence_start_diversity=round(start_div, 4),
            ngram_repetition_score=round(ngram, 4),
            punctuation_naturalness=round(punct, 2),
            flags=flags,
        )

    def score_batch(self, texts: list[str]) -> list[AuthenticityReport]:
        return [self.score(t) for t in texts]
