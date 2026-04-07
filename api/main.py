"""
Humanizer AI — FastAPI Inference Server

Endpoints:
  GET  /health              — Liveness + model status
  POST /generate            — Generate SEO content (supports streaming)
  POST /humanize            — Humanize existing AI-generated text
  POST /evaluate            — Score content across SEO + authenticity + readability
  POST /generate-and-eval   — Generate + auto-evaluate in one call

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.models import (
    GenerateRequest,
    GenerateResponse,
    HumanizeRequest,
    HumanizeResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
)
from api.inference import InferenceEngine
from evaluation.evaluator import ContentEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────

engine: InferenceEngine = None
evaluator: ContentEvaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, evaluator
    model_path = os.getenv("MODEL_PATH", "models/humanizer-llama3-8b-merged")
    use_4bit = os.getenv("USE_4BIT", "true").lower() == "true"

    logger.info(f"Loading inference engine from {model_path}")
    engine = InferenceEngine(
        model_path=model_path,
        use_4bit=use_4bit,
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "2048")),
    )
    engine.load()

    evaluator = ContentEvaluator(device="cpu")
    logger.info("Server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Humanizer AI API",
    description="QLoRA fine-tuned Llama-3 8B for SEO content humanization",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Request logging middleware ────────────────────────────────────────────────

class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
        return response

app.add_middleware(RequestTimingMiddleware)


# ── Dependency ────────────────────────────────────────────────────────────────

def require_model():
    if engine is None or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if (engine and engine.is_loaded) else "loading",
        model_loaded=engine.is_loaded if engine else False,
        model_id=os.getenv("MODEL_PATH", "models/humanizer-llama3-8b-merged"),
        device="cuda" if (engine and engine.is_loaded) else "unknown",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _engine: InferenceEngine = Depends(require_model),
):
    """Generate SEO-optimized, human-like content. Supports SSE streaming."""
    if req.stream:
        async def event_stream() -> AsyncGenerator[bytes, None]:
            async for token in _engine.generate_seo_content_streaming(
                keyword=req.keyword,
                niche=req.niche,
                content_format=req.content_format.value,
                target_word_count=req.target_word_count,
                tone=req.tone,
                secondary_keywords=req.secondary_keywords,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    content, elapsed_ms = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: _engine.generate_seo_content(
            keyword=req.keyword,
            niche=req.niche,
            content_format=req.content_format.value,
            target_word_count=req.target_word_count,
            tone=req.tone,
            secondary_keywords=req.secondary_keywords,
        ),
    )

    return GenerateResponse(
        content=content,
        word_count=len(content.split()),
        model_version="humanizer-llama3-8b-v1",
        generation_time_ms=int(elapsed_ms),
    )


@app.post("/humanize", response_model=HumanizeResponse)
async def humanize(
    req: HumanizeRequest,
    _engine: InferenceEngine = Depends(require_model),
):
    """Rewrite AI-generated text to be more human-like."""
    # Score before
    before_report = evaluator.authenticity.score(req.text)
    before_score = before_report.overall_score

    humanized, _ = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: _engine.humanize(req.text),
    )

    # Score after
    after_report = evaluator.authenticity.score(humanized)
    after_score = after_report.overall_score

    improvements = []
    if after_report.burstiness_score > before_report.burstiness_score:
        improvements.append("Improved sentence-length variation (burstiness)")
    if after_report.vocabulary_richness > before_report.vocabulary_richness:
        improvements.append("Richer vocabulary (higher TTR)")
    if after_report.sentence_start_diversity > before_report.sentence_start_diversity:
        improvements.append("More diverse sentence openers")
    if not improvements:
        improvements.append("Minor style and phrasing adjustments")

    return HumanizeResponse(
        original_text=req.text,
        humanized_text=humanized,
        before_score=before_score,
        after_score=after_score,
        improvements=improvements,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_content(req: EvaluateRequest):
    """Score content across SEO, authenticity, and readability metrics."""
    report = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: evaluator.evaluate(
            req.text,
            req.primary_keyword,
            req.secondary_keywords,
        ),
    )
    return EvaluateResponse(**report.to_dict())


@app.post("/generate-and-eval")
async def generate_and_evaluate(
    req: GenerateRequest,
    _engine: InferenceEngine = Depends(require_model),
):
    """
    Generate content then immediately evaluate it.
    Useful for pipeline automation where you want a single API call.
    """
    content, elapsed_ms = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: _engine.generate_seo_content(
            keyword=req.keyword,
            niche=req.niche,
            content_format=req.content_format.value,
            target_word_count=req.target_word_count,
            tone=req.tone,
            secondary_keywords=req.secondary_keywords,
        ),
    )

    eval_report = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: evaluator.evaluate(content, req.keyword, req.secondary_keywords),
    )

    return {
        "content": content,
        "word_count": len(content.split()),
        "generation_time_ms": int(elapsed_ms),
        "evaluation": eval_report.to_dict(),
        "model_version": "humanizer-llama3-8b-v1",
    }
