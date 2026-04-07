"""
Synthetic SEO Dataset Builder
Generates human-like SEO articles using GPT-4 with diverse writing styles,
then formats them into Alpaca/ShareGPT JSONL for QLoRA fine-tuning.
"""

import json
import random
import re
import time
import asyncio
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

# ── Writing style personas ────────────────────────────────────────────────────
WRITER_PERSONAS = [
    {
        "name": "experienced_blogger",
        "description": "A seasoned blogger with 10+ years writing about technology and business. Uses conversational tone, personal anecdotes, and occasional humor. Varies sentence length drastically—short punchy lines mixed with longer explanatory ones.",
    },
    {
        "name": "technical_journalist",
        "description": "A journalist who covers AI and tech. Writes with authority, uses real-world examples, interviews-style quotes, and data points. Tends toward active voice and concrete specifics.",
    },
    {
        "name": "subject_matter_expert",
        "description": "A domain expert writing to educate peers. Uses precise terminology, acknowledges nuance and trade-offs, includes caveats. Dense but accessible.",
    },
    {
        "name": "content_marketer",
        "description": "A content marketer focused on engagement and conversion. Uses power words, rhetorical questions, lists, and clear calls-to-action. Optimistic and solutions-oriented.",
    },
    {
        "name": "academic_practitioner",
        "description": "An academic who also works in industry. Balances evidence-based claims with practical advice. Cites implicit research, uses hedging language appropriately.",
    },
]

SEO_NICHES = [
    "SaaS tools", "digital marketing", "AI and machine learning", "e-commerce",
    "personal finance", "health and wellness", "travel", "cybersecurity",
    "developer tools", "content creation", "SEO strategies", "social media marketing",
    "B2B sales", "remote work productivity", "cloud computing",
]

ARTICLE_FORMATS = [
    "how-to guide", "listicle", "deep-dive analysis", "comparison article",
    "beginner's guide", "expert roundup", "case study", "opinion piece",
    "FAQ article", "ultimate guide",
]

HUMANIZATION_DIRECTIVES = [
    "Begin with a surprising statistic or counterintuitive claim.",
    "Include a brief personal anecdote or hypothetical scenario in the intro.",
    "Use at least two rhetorical questions throughout.",
    "Add a parenthetical aside that shows personality.",
    "Include one mildly self-deprecating or humble admission.",
    "Use a colloquial phrase or idiom naturally.",
    "Vary paragraph length: at least one single-sentence paragraph.",
    "Include a concrete real-world example with a brand or person's name.",
    "End a section with an open-ended thought or forward-looking statement.",
    "Use em-dashes for emphasis at least once.",
]


@dataclass
class ArticleConfig:
    niche: str
    keyword: str
    article_format: str
    persona: dict
    word_count_target: int
    humanization_directives: list[str]
    secondary_keywords: list[str] = field(default_factory=list)


@dataclass
class TrainingExample:
    instruction: str
    input: str
    output: str
    metadata: dict = field(default_factory=dict)


class SEODatasetBuilder:
    """
    Generates a synthetic dataset of SEO-optimized, human-like articles
    using GPT-4 with diverse personas and humanization techniques.
    Outputs Alpaca-format JSONL for QLoRA fine-tuning.
    """

    def __init__(self, api_key: str, output_dir: str = "data/raw"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_article_config(self, keyword: str) -> ArticleConfig:
        return ArticleConfig(
            niche=random.choice(SEO_NICHES),
            keyword=keyword,
            article_format=random.choice(ARTICLE_FORMATS),
            persona=random.choice(WRITER_PERSONAS),
            word_count_target=random.randint(800, 1800),
            humanization_directives=random.sample(HUMANIZATION_DIRECTIVES, k=4),
            secondary_keywords=self._generate_secondary_keywords(keyword),
        )

    def _generate_secondary_keywords(self, primary: str) -> list[str]:
        """Placeholder — in production, pull from SEMrush/Ahrefs API."""
        words = primary.lower().split()
        return [
            f"best {primary}",
            f"{primary} guide",
            f"how to use {primary}",
            f"{words[-1]} tips" if words else "tips",
        ]

    def _build_system_prompt(self, config: ArticleConfig) -> str:
        directives_block = "\n".join(
            f"  {i+1}. {d}" for i, d in enumerate(config.humanization_directives)
        )
        return f"""You are a {config.persona['name'].replace('_', ' ')} writing an SEO article.

PERSONA: {config.persona['description']}

YOUR WRITING RULES:
{directives_block}

SEO REQUIREMENTS:
- Primary keyword: "{config.keyword}" (use naturally 1–2% density, never stuff)
- Secondary keywords: {', '.join(f'"{k}"' for k in config.secondary_keywords)}
- Use H2/H3 headers with markdown (##, ###)
- Include a meta description comment at the top: <!-- meta: ... -->
- Target ~{config.word_count_target} words
- Format: {config.article_format}

CRITICAL: Write as a real human would. Vary sentence structure aggressively.
Never start consecutive sentences with the same word. Mix short and long sentences.
Show genuine expertise through specific details, not generic platitudes."""

    def _build_user_prompt(self, config: ArticleConfig) -> str:
        return (
            f"Write a {config.article_format} about \"{config.keyword}\" "
            f"for the {config.niche} niche. "
            f"Incorporate all writing rules and SEO requirements from the system prompt."
        )

    def _build_training_instruction(self, config: ArticleConfig) -> str:
        """The instruction that will be in the fine-tuning sample."""
        return (
            f"Write a high-quality, human-sounding SEO {config.article_format} "
            f"targeting the keyword \"{config.keyword}\" for the {config.niche} niche. "
            f"The content must read naturally, vary sentence structure, and incorporate "
            f"the keyword at approximately 1-2% density. Use markdown headers."
        )

    async def _generate_article(
        self, config: ArticleConfig, retries: int = 3
    ) -> Optional[str]:
        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self._build_system_prompt(config)},
                        {"role": "user", "content": self._build_user_prompt(config)},
                    ],
                    temperature=random.uniform(0.75, 0.95),
                    max_tokens=2500,
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                if attempt == retries - 1:
                    return None
        return None

    def _post_process(self, article: str) -> str:
        """Light cleanup — strip extra blank lines, normalize whitespace."""
        article = re.sub(r"\n{3,}", "\n\n", article)
        article = article.strip()
        return article

    def _to_training_example(
        self, article: str, config: ArticleConfig
    ) -> TrainingExample:
        return TrainingExample(
            instruction=self._build_training_instruction(config),
            input="",  # Alpaca format: empty input for pure generation task
            output=self._post_process(article),
            metadata={
                "keyword": config.keyword,
                "niche": config.niche,
                "format": config.article_format,
                "persona": config.persona["name"],
                "word_count": len(article.split()),
            },
        )

    async def generate_batch(
        self,
        keywords: list[str],
        concurrency: int = 5,
        output_file: str = "seo_dataset.jsonl",
    ) -> list[TrainingExample]:
        """
        Generate articles for all keywords with bounded concurrency.
        Saves incrementally to JSONL so progress is not lost on failure.
        """
        semaphore = asyncio.Semaphore(concurrency)
        output_path = self.output_dir / output_file
        examples: list[TrainingExample] = []

        async def _process(keyword: str) -> Optional[TrainingExample]:
            async with semaphore:
                config = self._build_article_config(keyword)
                article = await self._generate_article(config)
                if article:
                    example = self._to_training_example(article, config)
                    # Incremental write — crash-safe
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
                    return example
                return None

        tasks = [_process(kw) for kw in keywords]
        results = await tqdm_asyncio.gather(*tasks, desc="Generating articles")
        examples = [r for r in results if r is not None]
        logger.info(f"Generated {len(examples)}/{len(keywords)} articles → {output_path}")
        return examples

    def convert_to_sharegpt(self, alpaca_path: str, sharegpt_path: str) -> None:
        """
        Convert Alpaca-format JSONL → ShareGPT format for use with
        axolotl / LLaMA-Factory trainers that prefer multi-turn chat format.
        """
        sharegpt_records = []
        with open(alpaca_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompt = rec["instruction"]
                if rec.get("input"):
                    prompt += f"\n\n{rec['input']}"
                sharegpt_records.append({
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": rec["output"]},
                    ],
                    "metadata": rec.get("metadata", {}),
                })
        with open(sharegpt_path, "w", encoding="utf-8") as f:
            for rec in sharegpt_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Converted {len(sharegpt_records)} records to ShareGPT → {sharegpt_path}")

    def split_dataset(
        self,
        source_path: str,
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> tuple[str, str]:
        """Split into train/eval sets."""
        random.seed(seed)
        with open(source_path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        random.shuffle(records)
        split_idx = int(len(records) * train_ratio)
        train, eval_ = records[:split_idx], records[split_idx:]

        base = Path(source_path)
        train_path = str(base.parent / f"{base.stem}_train.jsonl")
        eval_path = str(base.parent / f"{base.stem}_eval.jsonl")

        for path, data in [(train_path, train), (eval_path, eval_)]:
            with open(path, "w", encoding="utf-8") as f:
                for rec in data:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"Split: {len(train)} train / {len(eval_)} eval")
        return train_path, eval_path


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build synthetic SEO fine-tuning dataset")
    parser.add_argument("--keywords-file", required=True, help="One keyword per line")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--output-file", default="seo_dataset.jsonl")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--convert-sharegpt", action="store_true")
    args = parser.parse_args()

    api_key = os.environ["OPENAI_API_KEY"]
    with open(args.keywords_file) as f:
        keywords = [l.strip() for l in f if l.strip()]

    builder = SEODatasetBuilder(api_key=api_key, output_dir=args.output_dir)

    async def main():
        await builder.generate_batch(
            keywords, concurrency=args.concurrency, output_file=args.output_file
        )
        out = str(Path(args.output_dir) / args.output_file)
        if args.convert_sharegpt:
            builder.convert_to_sharegpt(out, out.replace(".jsonl", "_sharegpt.jsonl"))
        builder.split_dataset(out)

    asyncio.run(main())
