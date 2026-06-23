#!/usr/bin/env python3
"""
Real-time SBD inference — usable both as a CLI demo and as an importable module.

Usage (CLI demo):
    python scripts/inference.py --model-dir runs/sbd_model/final
    python scripts/inference.py --model-dir runs/sbd_model/final --text "Tôi muốn đặt lịch"

Import usage:
    from scripts.inference import SBDPredictor
    predictor = SBDPredictor("runs/sbd_model/final")
    result = predictor.predict("Tôi muốn đặt lịch khám ngày mai")
    # → {"label": "complete", "confidence": 0.92, "is_complete": True}
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


class SBDPredictor:
    """Lightweight wrapper for real-time callbot inference."""

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.65,
        device: str | None = None,
        config_path: str = "config/config.yaml",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold

        # Try to load threshold from config if not overridden
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.threshold = threshold if threshold != 0.65 else cfg["inference"]["threshold"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Returns:
            {
              "label": "complete" | "incomplete",
              "is_complete": bool,
              "confidence": float,          # prob of predicted class
              "prob_complete": float,       # always the complete-class prob
              "latency_ms": float,
            }
        """
        t0 = time.perf_counter()
        enc = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        prob_incomplete, prob_complete = probs[0], probs[1]
        is_complete = prob_complete >= self.threshold
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "label": "complete" if is_complete else "incomplete",
            "is_complete": is_complete,
            "confidence": prob_complete if is_complete else prob_incomplete,
            "prob_complete": prob_complete,
            "latency_ms": round(latency_ms, 2),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch inference — more efficient for offline evaluation."""
        results = []
        t0 = time.perf_counter()
        enc = self.tokenizer(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()
        total_ms = (time.perf_counter() - t0) * 1000

        for i, (text, prob) in enumerate(zip(texts, probs)):
            prob_complete = prob[1]
            is_complete = prob_complete >= self.threshold
            results.append({
                "text": text,
                "label": "complete" if is_complete else "incomplete",
                "is_complete": is_complete,
                "prob_complete": prob_complete,
            })
        return results


# ── CLI interactive demo ──────────────────────────────────────────────────────

DEMO_EXAMPLES = [
    # complete
    ("Tôi muốn đặt lịch hẹn khám bệnh vào ngày mai.", 1),
    ("Cho tôi hỏi phí dịch vụ là bao nhiêu?", 1),
    ("Vâng, cảm ơn anh.", 1),
    ("Số điện thoại của tôi là 0912 345 678.", 1),
    ("Không, tôi không đồng ý mức đó.", 1),
    # incomplete
    ("Tôi muốn", 0),
    ("Ờ thì là", 0),
    ("Cho tôi hỏi về", 0),
    ("Vâng thì cái đó", 0),
    ("Ừm, số điện thoại của tôi là", 0),
]


def run_demo(predictor: SBDPredictor):
    print("\n" + "=" * 65)
    print(" SBD Demo — built-in test cases")
    print("=" * 65)
    correct = 0
    for text, true_label in DEMO_EXAMPLES:
        result = predictor.predict(text)
        pred = 1 if result["is_complete"] else 0
        ok = "✓" if pred == true_label else "✗"
        correct += pred == true_label
        print(
            f"{ok} [{result['label']:10s}] conf={result['prob_complete']:.3f} "
            f"({result['latency_ms']:.1f}ms)  "{text}""
        )
    print(f"\nAccuracy on demo cases: {correct}/{len(DEMO_EXAMPLES)}")

    print("\n" + "=" * 65)
    print(" Interactive mode — type a sentence, Ctrl-C to quit")
    print("=" * 65)
    while True:
        try:
            text = input("\n> ").strip()
            if not text:
                continue
            r = predictor.predict(text)
            print(
                f"  → {r['label'].upper()}  "
                f"(P(complete)={r['prob_complete']:.4f}, "
                f"latency={r['latency_ms']:.1f}ms)"
            )
        except KeyboardInterrupt:
            print("\nBye!")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to saved model directory")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--text", default=None, help="Single text to classify (no interactive mode)")
    args = parser.parse_args()

    predictor = SBDPredictor(args.model_dir, threshold=args.threshold, config_path=args.config)
    print(f"Model loaded | device={predictor.device} | threshold={predictor.threshold}")

    if args.text:
        result = predictor.predict(args.text)
        print(result)
    else:
        run_demo(predictor)


if __name__ == "__main__":
    main()
