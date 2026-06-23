#!/usr/bin/env python3
"""
Merge tất cả file JSONL trong data/raw/ và data/processed/ thành dataset cân bằng,
rồi split lại train/val/test.

Usage:
    python scripts/merge_and_split.py
    python scripts/merge_and_split.py --balance   # undersample class majority
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"text": row["text"], "label": row["label"]},
                                ensure_ascii=False) + "\n")
    c = Counter(r["label"] for r in rows)
    pct = 100 * c[1] / len(rows)
    print(f"  → {path.name}: {len(rows)} rows  complete={c[1]} ({pct:.1f}%)  incomplete={c[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", action="store_true",
                        help="Undersample majority class to 50/50")
    parser.add_argument("--val-ratio",  type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(__file__).parent.parent

    # ── Collect all available data ────────────────────────────────────────────
    sources = []
    for pattern in ["data/processed/*.jsonl", "data/raw/*.jsonl"]:
        sources.extend(base.glob(pattern))

    all_rows: list[dict] = []
    seen: set[str] = set()
    for src in sorted(sources):
        rows = load_jsonl(src)
        before = len(all_rows)
        for row in rows:
            key = row["text"].strip().lower()
            if key not in seen and row.get("label") in (0, 1):
                seen.add(key)
                all_rows.append({"text": row["text"], "label": row["label"]})
        print(f"Loaded {src.name}: {len(rows)} rows, {len(all_rows)-before} unique added")

    c = Counter(r["label"] for r in all_rows)
    print(f"\nTotal unique: {len(all_rows)}  complete={c[1]}  incomplete={c[0]}")

    # ── Balance (optional) ────────────────────────────────────────────────────
    if args.balance:
        min_class = min(c[0], c[1])
        complete   = [r for r in all_rows if r["label"] == 1]
        incomplete = [r for r in all_rows if r["label"] == 0]
        random.seed(args.seed)
        random.shuffle(complete)
        random.shuffle(incomplete)
        all_rows = complete[:min_class] + incomplete[:min_class]
        print(f"After balancing: {len(all_rows)} rows (50/50)")
    else:
        random.seed(args.seed)

    random.shuffle(all_rows)

    # ── Split ─────────────────────────────────────────────────────────────────
    n = len(all_rows)
    n_test = int(n * args.test_ratio)
    n_val  = int(n * args.val_ratio)
    test  = all_rows[:n_test]
    val   = all_rows[n_test:n_test + n_val]
    train = all_rows[n_test + n_val:]

    print(f"\nSplit: train={len(train)} | val={len(val)} | test={len(test)}")
    save_jsonl(train, base / "data/processed/train.jsonl")
    save_jsonl(val,   base / "data/processed/val.jsonl")
    save_jsonl(test,  base / "data/processed/test.jsonl")
    print("\nDone. Run: python scripts/train.py")


if __name__ == "__main__":
    main()
