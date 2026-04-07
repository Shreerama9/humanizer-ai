# Humanizer AI — QLoRA Fine-Tuning Pipeline for SEO Content

> Fine-tunes Llama-3 8B with QLoRA to generate human-like, brand-aligned SEO content that passes AI detection while maintaining ideal readability scores.

## Architecture

```
humanizer-ai/
├── data/
│   ├── dataset_builder.py     # GPT-4 powered synthetic SEO dataset generator
│   └── data_processor.py      # Filtering, formatting, tokenization (Alpaca → HF Dataset)
├── training/
│   ├── config.py              # Typed dataclass config for model/LoRA/training
│   ├── config.yaml            # YAML overrides for all training hyperparameters
│   ├── qlora_trainer.py       # 4-bit NF4 QLoRA pipeline (PEFT + TRL SFTTrainer)
│   └── train.py               # CLI entry point: train + merge adapter
├── evaluation/
│   ├── readability.py         # Flesch-Kincaid, Gunning Fog, SMOG (pure Python)
│   ├── authenticity_scorer.py # Multi-signal AI detection scorer (PPL, burstiness, TTR)
│   ├── seo_evaluator.py       # Keyword density, entity density, E-E-A-T, header structure
│   └── evaluator.py           # Unified pipeline: SEO 40% + Authenticity 35% + Readability 25%
└── api/
    ├── models.py              # Pydantic request/response schemas
    ├── inference.py           # 4-bit inference engine with streaming support
    └── main.py                # FastAPI server: /generate, /humanize, /evaluate
```

## Quickstart

### 1. Install dependencies
```bash
# Install PyTorch first — match your GPU architecture:
#   RTX 50xx Blackwell (sm_120): CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#   RTX 30xx / 40xx (sm_86/sm_89): CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
# Flash Attention (optional, requires CUDA 12.x):
pip install flash-attn --no-build-isolation
# spaCy NLP model for entity detection:
python -m spacy download en_core_web_sm
```

### 2. Generate synthetic training dataset
```bash
export OPENAI_API_KEY=sk-...
python -m data.dataset_builder \
  --keywords-file keywords.txt \
  --output-dir data/raw \
  --concurrency 5 \
  --convert-sharegpt
```
`keywords.txt` — one target SEO keyword per line.

### 3. Fine-tune with QLoRA
```bash
# Edit training/config.yaml to set paths and hyperparameters
python -m training.train --config training/config.yaml
```
Training logs stream to Weights & Biases. The LoRA adapter is saved to `checkpoints/`.

### 4. Merge adapter into base model
```bash
python -m training.train --merge-only \
  --adapter-path checkpoints/humanizer-llama3-8b
# Merged model saved to models/humanizer-llama3-8b-merged
```

### 5. Launch the inference API
```bash
export MODEL_PATH=models/humanizer-llama3-8b-merged
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## API Reference

### `POST /generate`
Generate SEO-optimized human-like content.

```json
{
  "keyword": "best project management tools",
  "niche": "SaaS tools",
  "content_format": "listicle",
  "secondary_keywords": ["task management software", "team collaboration tools"],
  "target_word_count": 1200,
  "tone": "conversational",
  "stream": false
}
```

**Response:**
```json
{
  "content": "# 11 Best Project Management Tools...",
  "word_count": 1187,
  "generation_time_ms": 4200,
  "model_version": "humanizer-llama3-8b-v1"
}
```

### `POST /humanize`
Rewrite AI-generated text to pass detection.

```json
{
  "text": "Project management tools are essential for...",
  "primary_keyword": "project management tools",
  "intensity": 0.8
}
```

### `POST /evaluate`
Score content quality across all dimensions.

```json
{
  "text": "...",
  "primary_keyword": "project management tools",
  "secondary_keywords": ["task management", "team productivity"]
}
```

**Response includes:**
- `composite_score` — weighted 0–100 (SEO 40%, Authenticity 35%, Readability 25%)
- `recommendation` — `PUBLISH` / `REVISE` / `REJECT`
- Full breakdowns per dimension

### `GET /health`
Liveness check and model status.

### `POST /generate-and-eval`
Single endpoint: generate + score in one call.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **QLoRA r=64** | High rank captures stylistic nuance; double quantization saves ~0.4 bits/param |
| **RSLoRA** | Rank-stabilized LoRA improves convergence at r≥64 without extra cost |
| **Target all 7 linear layers** | Including MLP gate/up/down projectors is critical for style, not just attention |
| **Response-only loss** | Only backprop on the output; prevents the model from overfitting to prompts |
| **Packing + group_by_length** | Maximizes GPU utilization; reduces padding waste by ~30% |
| **Multi-signal authenticity scorer** | Perplexity alone is gameable; burstiness + TTR + n-gram together are harder to fool |
| **Streaming via TextIteratorStreamer** | Background thread + sync queue; no special async model needed |

## Training Results (Reference Run)

| Metric | Before Fine-tuning | After Fine-tuning |
|---|---|---|
| Authenticity Score | 52.3 | 81.7 |
| Flesch Reading Ease | 48.2 | 67.4 |
| SEO Score (avg) | 61.0 | 78.3 |
| Composite Score | 54.0 | 76.1 |
| GPTZero Detection | 94% AI | 23% AI |

## Hardware Requirements

| Config | GPU | Training Time (1k samples, 3 epochs) |
|---|---|---|
| Minimum | RTX 3090 (24GB) | ~6 hours |
| Recommended | A100 40GB | ~2.5 hours |
| Optimal | A100 80GB / H100 | ~1.5 hours |

Inference: RTX 3080 (10GB) is sufficient with 4-bit quantization.
