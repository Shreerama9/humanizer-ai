"""Tests for the evaluation pipeline (no GPU / model required)."""

import pytest
from evaluation.readability import analyze as readability_analyze
from evaluation.authenticity_scorer import AuthenticityScorer
from evaluation.seo_evaluator import SEOEvaluator


SAMPLE_TEXT = (
    "Project management tools help teams stay organized and hit deadlines. "
    "Whether you're running a small startup or a large enterprise, having the right "
    "project management software makes a measurable difference. In this guide, we "
    "break down the best project management tools available today, covering features, "
    "pricing, and ideal use cases. Let's dive in."
) * 5  # ~300 words


def test_readability_returns_report():
    report = readability_analyze(SAMPLE_TEXT)
    assert hasattr(report, "flesch_reading_ease")
    assert 0 <= report.flesch_reading_ease <= 120
    assert hasattr(report, "gunning_fog")


def test_seo_evaluator_keyword_density():
    evaluator = SEOEvaluator()
    report = evaluator.evaluate(SAMPLE_TEXT, "project management tools")
    assert hasattr(report, "keyword_density")
    assert hasattr(report, "overall_seo_score")
    assert 0 <= report.overall_seo_score <= 100


def test_authenticity_scorer_no_model():
    """AuthenticityScorer without a PPL model should still score structural signals."""
    scorer = AuthenticityScorer(model_id=None, device="cpu")
    report = scorer.score(SAMPLE_TEXT)
    assert hasattr(report, "overall_score")
    assert 0 <= report.overall_score <= 100
