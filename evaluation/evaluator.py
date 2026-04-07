"""
Unified Content Evaluation Pipeline

Combines readability, authenticity, and SEO metrics into a single
pass/fail report. Used both during training (as eval callback) and
in the API response for real-time scoring.
"""

from dataclasses import dataclass, asdict
from typing import Optional

from evaluation.readability import analyze as readability_analyze, ReadabilityReport
from evaluation.authenticity_scorer import AuthenticityScorer, AuthenticityReport
from evaluation.seo_evaluator import SEOEvaluator, SEOReport


@dataclass
class ContentEvalReport:
    readability: ReadabilityReport
    authenticity: AuthenticityReport
    seo: SEOReport
    composite_score: float          # Weighted 0–100
    recommendation: str             # PUBLISH / REVISE / REJECT

    def to_dict(self) -> dict:
        return {
            "composite_score": self.composite_score,
            "recommendation": self.recommendation,
            "readability": asdict(self.readability),
            "authenticity": asdict(self.authenticity),
            "seo": asdict(self.seo),
        }


class ContentEvaluator:
    """
    Runs all three evaluation modules and produces a composite report.

    Weights:
      - SEO score:          40%
      - Authenticity score: 35%
      - Readability score:  25%
    """

    WEIGHTS = {"seo": 0.40, "authenticity": 0.35, "readability": 0.25}

    def __init__(
        self,
        ppl_model_id: Optional[str] = None,
        device: str = "cpu",
    ):
        self.readability = None          # Module is stateless
        self.authenticity = AuthenticityScorer(model_id=ppl_model_id, device=device)
        self.seo = SEOEvaluator()

    def _readability_score(self, report: ReadabilityReport) -> float:
        """Convert Flesch Reading Ease to 0–100 score targeting 60–80 range."""
        fre = report.flesch_reading_ease
        if 60 <= fre <= 80:
            return 100.0
        elif 50 <= fre < 60 or 80 < fre <= 90:
            return 75.0
        elif 40 <= fre < 50 or 90 < fre <= 100:
            return 50.0
        return 25.0

    def _recommendation(self, score: float) -> str:
        if score >= 75:
            return "PUBLISH"
        elif score >= 55:
            return "REVISE"
        return "REJECT"

    def evaluate(
        self,
        text: str,
        primary_keyword: str,
        secondary_keywords: Optional[list[str]] = None,
    ) -> ContentEvalReport:
        read_report = readability_analyze(text)
        auth_report = self.authenticity.score(text)
        seo_report = self.seo.evaluate(text, primary_keyword, secondary_keywords)

        read_score = self._readability_score(read_report)

        composite = (
            seo_report.overall_seo_score * self.WEIGHTS["seo"]
            + auth_report.overall_score * self.WEIGHTS["authenticity"]
            + read_score * self.WEIGHTS["readability"]
        )

        return ContentEvalReport(
            readability=read_report,
            authenticity=auth_report,
            seo=seo_report,
            composite_score=round(composite, 2),
            recommendation=self._recommendation(composite),
        )
