"""
Training configuration for QLoRA fine-tuning on Llama-3 8B.
All hyperparameters are grouped and documented for reproducibility.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # 4-bit NF4 quantization (QLoRA)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"          # NF4 outperforms FP4 on most tasks
    bnb_4bit_compute_dtype: str = "bfloat16"  # BF16 for stability on A100/H100
    bnb_4bit_use_double_quant: bool = True     # Nested quantization saves ~0.4 bits/param
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"  # Requires flash-attn>=2.0


@dataclass
class LoRAConfig:
    r: int = 64                    # Rank — higher = more capacity, more memory
    lora_alpha: int = 16           # Scale = alpha/r; keep low for stable training
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # MLP layers — critical for style
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = True        # Rank-stabilized LoRA — better convergence at high r


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints/humanizer-llama3-8b"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # Effective batch = 16
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"        # Memory-efficient AdamW
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = True                       # BF16 preferred on Ampere+ GPUs
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: list[str] = field(default_factory=lambda: ["wandb", "tensorboard"])
    run_name: str = "humanizer-qlora-llama3-8b"
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    group_by_length: bool = True            # Batch similar-length sequences → less padding
    packing: bool = True                    # Pack short sequences → better GPU utilization


@dataclass
class DataConfig:
    train_file: str = "data/raw/seo_dataset_train.jsonl"
    eval_file: str = "data/raw/seo_dataset_eval.jsonl"
    min_output_words: int = 200
    max_output_words: int = 2000


@dataclass
class MergeConfig:
    """Config for merging LoRA adapter back into base model for deployment."""
    adapter_path: str = "checkpoints/humanizer-llama3-8b"
    merged_output_dir: str = "models/humanizer-llama3-8b-merged"
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
