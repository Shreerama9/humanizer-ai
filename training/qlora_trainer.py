"""
QLoRA Fine-Tuning Pipeline — Llama-3 8B for SEO Content Humanization

Uses:
  - bitsandbytes 4-bit NF4 quantization (QLoRA)
  - PEFT LoRA adapters targeting all linear layers
  - TRL SFTTrainer with packing and gradient checkpointing
  - Weights & Biases for experiment tracking
  - Flash Attention 2 for memory-efficient attention
"""

import logging
import os
from pathlib import Path

import torch
import wandb
from datasets import DatasetDict
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from training.config import FullConfig
from data.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class HumanizerTrainer:
    """
    End-to-end QLoRA training pipeline for the SEO content humanizer.
    Handles model loading, LoRA wrapping, dataset preparation, and training.
    """

    def __init__(self, config: FullConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    # ── Model & tokenizer setup ───────────────────────────────────────────────

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_id,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="right",  # Required for SFT with packing
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _load_quantized_model(self):
        compute_dtype = getattr(torch, self.config.model.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.model.load_in_4bit,
            bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=self.config.model.attn_implementation,
            torch_dtype=compute_dtype,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        model.config.use_cache = False  # Disable KV cache during training
        model.config.pretraining_tp = 1  # Disable tensor parallelism for single-GPU
        return model

    def _apply_lora(self, model):
        # Prepare for k-bit training: casts LayerNorm and LM head to FP32
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.config.training.gradient_checkpointing,
        )
        lora_cfg = self.config.lora
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=lora_cfg.use_rslora,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, peft_config

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_training_args(self) -> TrainingArguments:
        tc = self.config.training
        return TrainingArguments(
            output_dir=tc.output_dir,
            num_train_epochs=tc.num_train_epochs,
            per_device_train_batch_size=tc.per_device_train_batch_size,
            per_device_eval_batch_size=tc.per_device_eval_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            gradient_checkpointing=tc.gradient_checkpointing,
            optim=tc.optim,
            learning_rate=tc.learning_rate,
            weight_decay=tc.weight_decay,
            lr_scheduler_type=tc.lr_scheduler_type,
            warmup_ratio=tc.warmup_ratio,
            fp16=tc.fp16,
            bf16=tc.bf16,
            logging_steps=tc.logging_steps,
            evaluation_strategy=tc.eval_strategy,
            eval_steps=tc.eval_steps,
            save_strategy=tc.save_strategy,
            save_steps=tc.save_steps,
            save_total_limit=tc.save_total_limit,
            load_best_model_at_end=tc.load_best_model_at_end,
            metric_for_best_model=tc.metric_for_best_model,
            greater_is_better=tc.greater_is_better,
            report_to=tc.report_to,
            run_name=tc.run_name,
            seed=tc.seed,
            dataloader_num_workers=tc.dataloader_num_workers,
            remove_unused_columns=tc.remove_unused_columns,
            group_by_length=tc.group_by_length,
        )

    def train(self, dataset: DatasetDict) -> None:
        logger.info("Loading tokenizer...")
        self.tokenizer = self._load_tokenizer()

        logger.info("Loading quantized model (4-bit NF4 QLoRA)...")
        self.model = self._load_quantized_model()

        logger.info("Applying LoRA adapters...")
        self.model, peft_config = self._apply_lora(self.model)

        training_args = self._build_training_args()

        # Response-only training: only compute loss on the ### Response: part
        response_template = "### Response:\n"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("eval"),
            peft_config=peft_config,
            max_seq_length=self.config.training.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=self.config.training.packing,
            data_collator=collator,
            callbacks=[LoggingCallback()],
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info(f"Saving adapter to {self.config.training.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.training.output_dir)

    # ── Merge LoRA → base model ───────────────────────────────────────────────

    def merge_and_export(self) -> None:
        """
        Merge LoRA weights back into the base model for efficient inference.
        The merged model can be loaded without PEFT at serving time.
        """
        from peft import AutoPeftModelForCausalLM

        logger.info("Loading adapter for merge...")
        merged_model = AutoPeftModelForCausalLM.from_pretrained(
            self.config.merge.adapter_path,
            device_map="cpu",                # Merge on CPU to avoid OOM
            torch_dtype=torch.float16,
        )
        merged_model = merged_model.merge_and_unload()

        out_dir = self.config.merge.merged_output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")

        tokenizer = AutoTokenizer.from_pretrained(self.config.merge.adapter_path)
        tokenizer.save_pretrained(out_dir)

        logger.info(f"Merged model saved to {out_dir}")

        if self.config.merge.push_to_hub and self.config.merge.hub_repo_id:
            merged_model.push_to_hub(self.config.merge.hub_repo_id)
            tokenizer.push_to_hub(self.config.merge.hub_repo_id)
            logger.info(f"Pushed to Hub: {self.config.merge.hub_repo_id}")


class LoggingCallback(TrainerCallback):
    """Logs train/eval metrics to W&B at each evaluation step."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        if wandb.run:
            wandb.log(
                {f"eval/{k}": v for k, v in metrics.items()},
                step=state.global_step,
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict,
        **kwargs,
    ):
        logger.info(f"Step {state.global_step}: {logs}")
