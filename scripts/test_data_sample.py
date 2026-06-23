#!/usr/bin/env python3
"""
Quick test: generate a small batch (20 examples) and print them.
Use this to verify OpenAI data quality before a full generation run.

Usage:
    python scripts/test_data_sample.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.generate import generate_batch
from openai import OpenAI
import os


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    scenario = "đặt lịch hẹn khám bệnh"

    print(f"Generating 20 test samples for scenario: '{scenario}'\n")
    examples = generate_batch(client, scenario, n=20, model="gpt-4o-mini", temperature=0.9)

    n_complete = sum(1 for e in examples if e["label"] == 1)
    print(f"Got {len(examples)} examples | complete={n_complete} | incomplete={len(examples)-n_complete}\n")

    print("─" * 60)
    for ex in sorted(examples, key=lambda e: e["label"]):
        label_str = "COMPLETE  " if ex["label"] == 1 else "INCOMPLETE"
        print(f"[{label_str}]  {ex['text']}")
    print("─" * 60)
    print("\nIf quality looks good, run: python scripts/generate_data.py")


if __name__ == "__main__":
    main()
