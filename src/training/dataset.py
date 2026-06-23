"""PyTorch Dataset for SBD binary classification."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SBDDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 128,
    ):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        # PhoBERT expects pre-tokenized text with spaces between syllables;
        # other tokenizers handle raw text fine. We pass raw text here —
        # PhoBERT's AutoTokenizer from HuggingFace handles it correctly.
        encoding = self.tokenizer(
            ex["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }
        # token_type_ids not used by PhoBERT (RoBERTa-based); include only if present
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item

    def label_counts(self) -> dict:
        from collections import Counter
        counts = Counter(e["label"] for e in self.examples)
        return dict(counts)
