"""
Main training entry point.

Usage:
  python -m training.train --config config.yaml
  python -m training.train --merge-only --adapter-path checkpoints/humanizer-llama3-8b
"""

import argparse
import logging
import os
from dataclasses import asdict

import wandb
import yaml

from training.config import FullConfig, ModelConfig, LoRAConfig, TrainingConfig, DataConfig, MergeConfig
from training.qlora_trainer import HumanizerTrainer
from data.data_processor import DataProcessor
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> FullConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = FullConfig(
        model=ModelConfig(**raw.get("model", {})),
        lora=LoRAConfig(**raw.get("lora", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        merge=MergeConfig(**raw.get("merge", {})),
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml", help="YAML config path")
    parser.add_argument("--merge-only", action="store_true", help="Skip training, only merge adapter")
    parser.add_argument("--no-merge", action="store_true", help="Skip merge step after training")
    parser.add_argument("--adapter-path", help="Override adapter path for merge")
    parser.add_argument("--resume", metavar="CHECKPOINT_DIR", help="Resume training from a checkpoint directory")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else FullConfig()

    if args.adapter_path:
        config.merge.adapter_path = args.adapter_path

    if args.no_wandb:
        config.training.report_to = ["tensorboard"]
    else:
        wandb.init(
            project="humanizer-ai",
            name=config.training.run_name,
            config={
                "model": asdict(config.model),
                "lora": asdict(config.lora),
                "training": asdict(config.training),
            },
        )

    trainer = HumanizerTrainer(config)

    if not args.merge_only:
        # Load tokenizer early for preprocessing
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_id, padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        processor = DataProcessor(
            tokenizer=tokenizer,
            max_seq_length=config.training.max_seq_length,
            min_output_words=config.data.min_output_words,
            max_output_words=config.data.max_output_words,
        )
        dataset = processor.load_and_process(
            train_path=config.data.train_file,
            eval_path=config.data.eval_file,
            tokenize=False,  # SFTTrainer handles tokenization with packing
        )
        logger.info(f"Dataset: {dataset}")
        trainer.train(dataset, resume_from_checkpoint=args.resume)

    if not args.no_merge:
        logger.info("Merging LoRA adapter into base model...")
        trainer.merge_and_export()
    else:
        logger.info("Skipping merge (--no-merge specified).")

    if wandb.run:
        wandb.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
