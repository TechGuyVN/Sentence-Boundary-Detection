#!/usr/bin/env python3
"""
Generate synthetic Vietnamese SBD data using OpenAI.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --total 5000 --model gpt-4o-mini
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.generate import generate_dataset, split_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--total", type=int, default=None, help="Override total samples")
    parser.add_argument("--model", default=None, help="Override OpenAI model")
    parser.add_argument("--no-split", action="store_true", help="Save all to data/raw/all.jsonl")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gen_cfg = cfg["generation"]
    data_cfg = cfg["data"]

    total = args.total or gen_cfg["n_samples_train"]
    model = args.model or gen_cfg["openai_model"]
    scenarios = gen_cfg["scenarios"]

    print(f"Generating {total} samples using {model}...")
    print(f"Scenarios ({len(scenarios)}): {', '.join(scenarios[:3])}...")

    all_examples = list(
        generate_dataset(
            scenarios=scenarios,
            total_samples=total,
            batch_size=gen_cfg["batch_size"],
            model=model,
            temperature=gen_cfg["temperature"],
        )
    )

    if args.no_split:
        from src.data_generation.generate import save_jsonl
        save_jsonl(all_examples, "data/raw/all.jsonl")
    else:
        split_and_save(
            all_examples,
            train_path=data_cfg["train_file"],
            val_path=data_cfg["val_file"],
            test_path=data_cfg["test_file"],
        )


if __name__ == "__main__":
    main()
