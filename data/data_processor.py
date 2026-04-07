"""
Data Processor — cleans, filters, and tokenizes the raw JSONL dataset
before feeding it into the QLoRA trainer.
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Prompt template — matches what we'll use at inference time
ALPACA_PROMPT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


class DataProcessor:
    """
    Loads raw Alpaca-format JSONL, applies quality filters,
    and produces tokenized HuggingFace Datasets ready for SFTTrainer.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 2048,
        min_output_words: int = 200,
        max_output_words: int = 2000,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_output_words = min_output_words
        self.max_output_words = max_output_words

    # ── Quality filters ───────────────────────────────────────────────────────

    def _is_valid(self, rec: dict) -> bool:
        output = rec.get("output", "")
        word_count = len(output.split())
        if word_count < self.min_output_words:
            return False
        if word_count > self.max_output_words:
            return False
        # Reject if output is mostly repetitive (sign of degenerate generation)
        if self._repetition_ratio(output) > 0.4:
            return False
        # Must have at least one markdown header
        if not re.search(r"^#{1,3}\s", output, re.MULTILINE):
            return False
        return True

    def _repetition_ratio(self, text: str, window: int = 8) -> float:
        """Fraction of n-gram windows that are duplicates — detects looping output."""
        words = text.lower().split()
        if len(words) < window * 2:
            return 0.0
        ngrams = [tuple(words[i:i+window]) for i in range(len(words) - window)]
        return 1.0 - len(set(ngrams)) / len(ngrams)

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_prompt(self, rec: dict) -> str:
        if rec.get("input", "").strip():
            return ALPACA_PROMPT.format(
                instruction=rec["instruction"],
                input=rec["input"],
                output=rec["output"],
            )
        return ALPACA_PROMPT_NO_INPUT.format(
            instruction=rec["instruction"],
            output=rec["output"],
        )

    def _tokenize(self, prompt: str) -> Optional[dict]:
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        # Skip samples that got truncated significantly (>10% loss)
        approx_full_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if approx_full_len > self.max_seq_length * 1.1:
            return None
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    # ── Public API ────────────────────────────────────────────────────────────

    def load_and_process(
        self,
        train_path: str,
        eval_path: Optional[str] = None,
        tokenize: bool = True,
    ) -> DatasetDict:
        def _load(path: str) -> list[dict]:
            records = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if self._is_valid(rec):
                            records.append({"text": self._format_prompt(rec)})
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Loaded {len(records)} valid examples from {path}")
            return records

        splits = {"train": Dataset.from_list(_load(train_path))}
        if eval_path:
            splits["eval"] = Dataset.from_list(_load(eval_path))

        if tokenize:
            def _tok(batch):
                return self.tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                )
            splits = {
                k: v.map(_tok, batched=True, remove_columns=["text"])
                for k, v in splits.items()
            }

        return DatasetDict(splits)

    def get_formatting_func(self):
        """
        Returns a formatting function compatible with TRL's SFTTrainer.

        SFTTrainer calls this with a batch dict ({"instruction": [...], "output": [...], ...})
        and expects a list of formatted strings back.
        Note: when using DataProcessor.load_and_process(), the dataset already has a
        "text" field — pass dataset_text_field="text" to SFTTrainer instead of this func.
        Use this only if you load raw Alpaca JSONL without pre-formatting.
        """
        def formatting_prompts_func(batch: dict) -> list[str]:
            instructions = batch.get("instruction", [])
            inputs = batch.get("input", [""] * len(instructions))
            outputs = batch.get("output", [""] * len(instructions))
            return [
                self._format_prompt({"instruction": inst, "input": inp, "output": out})
                for inst, inp, out in zip(instructions, inputs, outputs)
            ]
        return formatting_prompts_func
