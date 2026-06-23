#!/usr/bin/env python3
"""
Loại bỏ dấu câu khỏi tất cả training data.
ASR không sinh dấu câu nên model không nên học pattern từ dấu câu.

Usage:
    python scripts/clean_punctuation.py              # preview + fix data/
    python scripts/clean_punctuation.py --dry-run    # chỉ xem không ghi
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter


# Dấu câu cần xóa — giữ nguyên dấu thanh tiếng Việt (ă â đ ê ô ơ ư + tones)
_PUNCT_RE = re.compile(r"[.!?,;:\"'()\[\]{}\-–—…/\\|<>@#$%^&*+=~`]")


def clean_text(text: str) -> str:
    """Strip punctuation, normalize whitespace."""
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def process_file(path: Path, dry_run: bool) -> dict:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    cleaned, changed = [], 0
    for row in rows:
        orig = row["text"]
        new  = clean_text(orig)
        if orig != new:
            changed += 1
        cleaned.append({"text": new, "label": row["label"]})

    if not dry_run and changed > 0:
        with open(path, "w", encoding="utf-8") as f:
            for row in cleaned:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"total": len(rows), "changed": changed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    targets = list((base / "data" / "processed").glob("*.jsonl")) + \
              list((base / "data" / "raw").glob("*.jsonl"))

    print(f"{'DRY RUN — ' if args.dry_run else ''}Cleaning punctuation from training data\n")
    total_changed = 0
    for path in sorted(targets):
        stats = process_file(path, args.dry_run)
        total_changed += stats["changed"]
        action = "would change" if args.dry_run else "changed"
        print(f"  {path.name:<25} {stats['total']:>5} rows | {action}: {stats['changed']}")

    print(f"\nTotal rows {('that would be ' if args.dry_run else '')}modified: {total_changed}")
    if not args.dry_run:
        print("Done. Run: python scripts/merge_and_split.py --balance && python scripts/train.py")


if __name__ == "__main__":
    main()
