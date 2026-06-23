"""
Singleton model loader — loaded once at startup, reused for all requests.
Thread-safe for multi-worker deployments (each worker loads its own copy).
"""

import os
import re
import time
import logging
from pathlib import Path

# ASR không sinh dấu câu — strip trước khi inference để tránh bias từ model
_PUNCT_STRIP_RE = re.compile(r"[.!?,;:\"'()\[\]{}\-–—…]")

def clean_asr_text(text: str) -> str:
    text = _PUNCT_STRIP_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# ── Từ cuối câu yêu cầu bổ ngữ → buộc INCOMPLETE ───────────────────────────
_DANGLING_RE = re.compile(
    r"""(?ix)\b(
        cũng | thì | là | và | hoặc | nhưng | mà | để | vì |
        bởi  | rằng | hay | nếu | khi | sẽ | bị | nên | vẫn | chỉ |
        tuy\s+nhiên | trong\s+vòng | bởi\s+vì | chính\s+là
    )\s*[.!?,;]?\s*$""",
    re.UNICODE,
)

# ── Từ/cụm ngắn hoàn toàn COMPLETE — model thường đánh giá thiếu tự tin ─────
# (xã giao 1 từ, đồng ý ngắn, kết thúc lịch sự)
_ACK_RE = re.compile(
    r"""(?ix)^\s*(
        dạ | vâng | ừ | ok | okay | rồi | thôi | xong |
        được | cũng\s+được | thôi\s+được | ừ\s+được | ừ\s+cũng\s+được |
        dạ\s+rồi | vâng\s+rồi | ok\s+rồi | thôi\s+thì\s+thôi |
        không\s+cần | không\s+sao | không\s+cần\s+đâu
    )\s*[.!,]?\s*$""",
    re.UNICODE,
)


class SBDPredictor:
    def __init__(self, model_dir: str, threshold: float, max_len: int = 128):
        self.model_dir  = model_dir
        self.threshold  = threshold
        self.max_len    = max_len
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    def _load(self):
        t0 = time.perf_counter()
        logger.info("Loading tokenizer and model from %s …", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.model.to(self.device)
        elapsed = round((time.perf_counter() - t0) * 1000)
        logger.info("Model loaded in %dms on %s", elapsed, self.device)

    @staticmethod
    def _dangling_override(text: str) -> bool:
        """Return True nếu câu kết thúc bằng từ cần bổ ngữ → buộc INCOMPLETE."""
        return bool(_DANGLING_RE.search(text.strip()))

    def _apply_overrides(self, text: str, p_complete: float, p_incomplete: float):
        """Apply post-processing rules, return (p_complete, override_tag | None)."""
        if _ACK_RE.match(text):
            return max(p_complete, self.threshold + 0.01), "ack_word"
        if p_complete >= self.threshold and _DANGLING_RE.search(text.strip()):
            return min(p_incomplete, self.threshold - 0.01), "dangling_word"
        return p_complete, None

    @torch.no_grad()
    def predict_one(self, text: str) -> dict:
        t_start = time.perf_counter()

        text = clean_asr_text(text)

        # ── Tokenize ──────────────────────────────────────────────────────────
        t_tok = time.perf_counter()
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenize_ms = round((time.perf_counter() - t_tok) * 1000, 2)

        # ── Model inference ───────────────────────────────────────────────────
        t_inf = time.perf_counter()
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = F.softmax(logits, dim=-1).squeeze().tolist()
        inference_ms = round((time.perf_counter() - t_inf) * 1000, 2)

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        p_complete, overridden = self._apply_overrides(text, probs[1], probs[0])

        result = {
            "text":            text,
            "label":           "complete" if p_complete >= self.threshold else "incomplete",
            "is_complete":     p_complete >= self.threshold,
            "prob_complete":   round(p_complete, 4),
            "prob_incomplete": round(1 - p_complete, 4),
            "latency": {
                "tokenize_ms":  tokenize_ms,
                "inference_ms": inference_ms,
                "total_ms":     total_ms,
            },
        }
        if overridden:
            result["override"] = overridden
        return result

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> dict:
        t_start = time.perf_counter()

        texts = [clean_asr_text(t) for t in texts]

        # ── Tokenize ──────────────────────────────────────────────────────────
        t_tok = time.perf_counter()
        enc = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenize_ms = round((time.perf_counter() - t_tok) * 1000, 2)

        # ── Model inference ───────────────────────────────────────────────────
        t_inf = time.perf_counter()
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = F.softmax(logits, dim=-1).tolist()
        inference_ms = round((time.perf_counter() - t_inf) * 1000, 2)

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        results = []
        for text, p in zip(texts, probs):
            p_complete, overridden = self._apply_overrides(text, p[1], p[0])
            item = {
                "text":            text,
                "label":           "complete" if p_complete >= self.threshold else "incomplete",
                "is_complete":     p_complete >= self.threshold,
                "prob_complete":   round(p_complete, 4),
                "prob_incomplete": round(p[0], 4),
            }
            if overridden:
                item["override"] = overridden
            results.append(item)

        return {
            "results": results,
            "latency": {
                "tokenize_ms":  tokenize_ms,
                "inference_ms": inference_ms,
                "total_ms":     total_ms,
                "per_item_ms":  round(total_ms / len(texts), 2),
            },
        }


# Module-level singleton — created when the FastAPI app starts
_predictor: SBDPredictor | None = None


def get_predictor() -> SBDPredictor:
    return _predictor


def init_predictor(model_dir: str, threshold: float) -> SBDPredictor:
    global _predictor
    _predictor = SBDPredictor(model_dir, threshold)
    return _predictor
