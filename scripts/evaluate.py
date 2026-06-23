#!/usr/bin/env python3
"""
Evaluate a trained SBD model on the test set.

Usage:
    python scripts/evaluate.py --model-dir runs/sbd_model/final
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataset import SBDDataset
from src.utils.metrics import compute_metrics, full_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to saved model directory")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    file_map = {"test": data_cfg["test_file"], "val": data_cfg["val_file"], "train": data_cfg["train_file"]}
    data_file = file_map[args.split]

    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    print(f"Loading {args.split} data from: {data_file}")
    dataset = SBDDataset(data_file, tokenizer, model_cfg["max_seq_len"])
    print(f"  {len(dataset)} examples | label dist: {dataset.label_counts()}")

    training_args = TrainingArguments(
        output_dir="/tmp/sbd_eval",
        per_device_eval_batch_size=64,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    print("\nRunning inference...")
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=-1)

    print("\n" + "=" * 60)
    print(f"Results on [{args.split}] split:")
    print("=" * 60)
    print(full_report(labels.tolist(), preds.tolist()))
    print(f"\nAggregate metrics: {predictions.metrics}")


if __name__ == "__main__":
    main()
