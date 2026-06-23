#!/usr/bin/env python3
"""
Train the SBD binary classifier.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/config.yaml --epochs 3
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset import SBDDataset
from src.utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    set_seed(train_cfg["seed"])

    # ── Model & tokenizer ────────────────────────────────────────────────────
    model_name = model_cfg["name"]
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=model_cfg["num_labels"],
        hidden_dropout_prob=model_cfg.get("dropout", 0.1),
        attention_probs_dropout_prob=model_cfg.get("dropout", 0.1),
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    max_len = model_cfg["max_seq_len"]
    print("Loading datasets...")
    train_ds = SBDDataset(data_cfg["train_file"], tokenizer, max_len)
    val_ds = SBDDataset(data_cfg["val_file"], tokenizer, max_len)
    print(f"  train={len(train_ds)} | val={len(val_ds)}")
    print(f"  train label dist: {train_ds.label_counts()}")

    # ── Training args ─────────────────────────────────────────────────────────
    use_fp16 = train_cfg.get("fp16", False) and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=args.epochs or train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        learning_rate=float(args.lr or train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        fp16=use_fp16,
        logging_steps=50,
        report_to="none",
        seed=train_cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting training on {'GPU' if use_fp16 else 'CPU'}...")
    trainer.train()

    print("\nEvaluating on validation set...")
    results = trainer.evaluate()
    print(f"Val results: {results}")

    # Save final model
    save_path = f"{train_cfg['output_dir']}/final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
