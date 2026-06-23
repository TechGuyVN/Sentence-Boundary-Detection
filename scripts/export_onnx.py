#!/usr/bin/env python3
"""
Export trained model to ONNX for fast production inference.

Usage:
    python scripts/export_onnx.py --model-dir runs/sbd_model/final --output exports/sbd.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def export(model_dir: str, output_path: str, max_len: int = 128, opset: int = 17):
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy = tokenizer("Tôi muốn đặt lịch", max_length=max_len, padding="max_length",
                      truncation=True, return_tensors="pt")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to ONNX (opset={opset})...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    print(f"Exported → {output_path}")

    # Quick sanity check
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    inputs = {
        "input_ids": dummy["input_ids"].numpy(),
        "attention_mask": dummy["attention_mask"].numpy(),
    }
    out = sess.run(["logits"], inputs)[0]
    import torch.nn.functional as F
    probs = F.softmax(torch.tensor(out), dim=-1).numpy()
    print(f"Sanity check OK — prob_complete={probs[0][1]:.4f}")
    print(f"\nTo use ONNX in production:")
    print(f"  import onnxruntime as ort")
    print(f"  sess = ort.InferenceSession('{output_path}')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="exports/sbd.onnx")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.model_dir, args.output, args.max_len, args.opset)


if __name__ == "__main__":
    main()
