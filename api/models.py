"""Pydantic request/response models for the Humanizer AI API."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ContentFormat(str, Enum):
    HOW_TO = "how-to guide"
    LISTICLE = "listicle"
    DEEP_DIVE = "deep-dive analysis"
    COMPARISON = "comparison article"
    BEGINNERS_GUIDE = "beginner's guide"
    CASE_STUDY = "case study"
    OPINION = "opinion piece"
    FAQ = "FAQ article"
    ULTIMATE_GUIDE = "ultimate guide"


class GenerateRequest(BaseModel):
    keyword: str = Field(..., min_length=2, max_length=200, description="Primary SEO keyword")
    niche: str = Field(..., min_length=2, max_length=100, description="Content niche/topic area")
    content_format: ContentFormat = ContentFormat.HOW_TO
    secondary_keywords: list[str] = Field(default_factory=list, max_length=10)
    target_word_count: int = Field(default=1000, ge=300, le=3000)
    brand_name: Optional[str] = Field(None, max_length=100)
    tone: str = Field(default="informative", description="e.g. conversational, authoritative, casual")
    stream: bool = Field(default=False, description="Enable SSE streaming response")

    @field_validator("secondary_keywords")
    @classmethod
    def validate_secondary_kws(cls, v):
        return [kw.strip() for kw in v if kw.strip()]


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=10000, description="AI-generated text to humanize")
    primary_keyword: str = Field(..., min_length=2, max_length=200)
    intensity: float = Field(default=0.7, ge=0.0, le=1.0, description="Humanization strength 0–1")
    preserve_structure: bool = Field(default=True, description="Keep original headers/sections")


class EvaluateRequest(BaseModel):
    text: str = Field(..., min_length=50)
    primary_keyword: str = Field(..., min_length=2, max_length=200)
    secondary_keywords: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    content: str
    word_count: int
    evaluation: Optional[dict] = None
    model_version: str
    generation_time_ms: int


class HumanizeResponse(BaseModel):
    original_text: str
    humanized_text: str
    before_score: float
    after_score: float
    improvements: list[str]


class EvaluateResponse(BaseModel):
    composite_score: float
    recommendation: str
    readability: dict
    authenticity: dict
    seo: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str
    version: str = "1.0.0"
