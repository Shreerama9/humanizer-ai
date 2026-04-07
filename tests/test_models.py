"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from api.models import (
    ContentFormat,
    GenerateRequest,
    HumanizeRequest,
    EvaluateRequest,
)


def test_generate_request_defaults():
    req = GenerateRequest(keyword="best SEO tools", niche="digital marketing")
    assert req.content_format == ContentFormat.HOW_TO
    assert req.target_word_count == 1000
    assert req.tone == "informative"
    assert req.stream is False


def test_generate_request_strips_secondary_keywords():
    req = GenerateRequest(
        keyword="test keyword",
        niche="tech",
        secondary_keywords=["  kw1 ", "kw2", "  "],
    )
    assert req.secondary_keywords == ["kw1", "kw2"]


def test_generate_request_word_count_bounds():
    with pytest.raises(ValidationError):
        GenerateRequest(keyword="test", niche="tech", target_word_count=100)
    with pytest.raises(ValidationError):
        GenerateRequest(keyword="test", niche="tech", target_word_count=5000)


def test_humanize_request_min_length():
    with pytest.raises(ValidationError):
        HumanizeRequest(text="short", primary_keyword="kw")


def test_evaluate_request_valid():
    req = EvaluateRequest(
        text="A" * 100,
        primary_keyword="test keyword",
        secondary_keywords=["kw1"],
    )
    assert req.primary_keyword == "test keyword"


def test_content_format_values():
    assert ContentFormat.LISTICLE == "listicle"
    assert ContentFormat.HOW_TO == "how-to guide"
