"""
Model inference engine — handles loading the fine-tuned model,
batched generation, streaming, and the humanization rewrite pass.
"""

import logging
import re
import time
from typing import AsyncGenerator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert SEO content writer. Write in a natural, human voice. "
    "Vary sentence length. Use specific examples. Sound genuine, not corporate."
)

GENERATE_PROMPT = """### Instruction:
Write a high-quality, human-sounding SEO {format} targeting the keyword "{keyword}" for the {niche} niche.
The content must read naturally, vary sentence structure, and incorporate the keyword at approximately 1-2% density.
Use markdown headers. Target approximately {word_count} words.
{tone_instruction}
{secondary_kw_instruction}

### Response:
"""

HUMANIZE_PROMPT = """### Instruction:
Rewrite the following AI-generated text to sound more human and natural, while preserving all key information, SEO keywords, and structure. Make it feel like it was written by an experienced human writer — vary sentence length, add natural transitions, remove robotic phrasing, and inject personality where appropriate.

Original text:
{text}

### Response:
"""


class InferenceEngine:
    """
    Wraps a fine-tuned (or merged) Llama-3 model for inference.
    Supports 4-bit quantized loading for memory-efficient serving.
    """

    def __init__(
        self,
        model_path: str,
        use_4bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 2048,
    ):
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._loaded = False

    def load(self):
        logger.info(f"Loading model from {self.model_path} (4bit={self.use_4bit})")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype=torch.float16,
            )

        self._model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _build_generate_prompt(
        self,
        keyword: str,
        niche: str,
        content_format: str,
        target_word_count: int,
        tone: str = "informative",
        secondary_keywords: Optional[list[str]] = None,
    ) -> str:
        tone_instruction = f"Tone: {tone}." if tone else ""
        sec_kw_instruction = ""
        if secondary_keywords:
            sec_kw_instruction = f"Also incorporate these related keywords naturally: {', '.join(secondary_keywords)}."

        return GENERATE_PROMPT.format(
            format=content_format,
            keyword=keyword,
            niche=niche,
            word_count=target_word_count,
            tone_instruction=tone_instruction,
            secondary_kw_instruction=sec_kw_instruction,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        temperature: float = 0.75,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def generate_streaming(
        self,
        prompt: str,
        temperature: float = 0.75,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
    ) -> AsyncGenerator[str, None]:
        """Yields tokens as they are generated for SSE streaming."""
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._model.device)

        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        # Run generation in a background thread
        thread = Thread(target=self._model.generate, kwargs=generate_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    def generate_seo_content(
        self,
        keyword: str,
        niche: str,
        content_format: str = "how-to guide",
        target_word_count: int = 1000,
        tone: str = "informative",
        secondary_keywords: Optional[list[str]] = None,
    ) -> tuple[str, float]:
        """Returns (generated_text, generation_time_ms)."""
        prompt = self._build_generate_prompt(
            keyword, niche, content_format, target_word_count,
            tone, secondary_keywords,
        )
        start = time.perf_counter()
        content = self.generate(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return content, elapsed_ms

    def humanize(self, text: str) -> tuple[str, float]:
        """Runs a humanization rewrite pass on existing AI text."""
        prompt = HUMANIZE_PROMPT.format(text=text[:3000])  # Limit input length
        start = time.perf_counter()
        result = self.generate(prompt, temperature=0.85, repetition_penalty=1.1)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    async def generate_seo_content_streaming(
        self,
        keyword: str,
        niche: str,
        content_format: str = "how-to guide",
        target_word_count: int = 1000,
        tone: str = "informative",
        secondary_keywords: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        prompt = self._build_generate_prompt(
            keyword, niche, content_format, target_word_count,
            tone, secondary_keywords,
        )
        async for token in self.generate_streaming(prompt):
            yield token
