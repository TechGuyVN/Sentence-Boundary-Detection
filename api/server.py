"""
SBD Prediction API
Run: uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import json as _json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, field_validator


class UTF8JSONResponse(Response):
    """JSONResponse that outputs proper UTF-8 (not ASCII-escaped unicode)."""
    media_type = "application/json; charset=utf-8"

    def render(self, content) -> bytes:
        return _json.dumps(content, ensure_ascii=False).encode("utf-8")

from api.predictor import get_predictor, init_predictor

# ── Config từ env vars ────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = os.getenv("SBD_MODEL_DIR",  str(BASE_DIR / "runs/sbd_model/final"))
THRESHOLD = float(os.getenv("SBD_THRESHOLD", "0.65"))
MAX_BATCH = int(os.getenv("SBD_MAX_BATCH", "32"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sbd.api")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class BatchRequest(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def check_batch(cls, v: list[str]) -> list[str]:
        v = [t.strip() for t in v if t.strip()]
        if not v:
            raise ValueError("texts must contain at least one non-empty string")
        if len(v) > MAX_BATCH:
            raise ValueError(f"batch size exceeds limit ({MAX_BATCH})")
        return v


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SBD API — model=%s  threshold=%.2f", MODEL_DIR, THRESHOLD)
    if not Path(MODEL_DIR).exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")
    init_predictor(MODEL_DIR, THRESHOLD)
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SBD — Sentence Boundary Detection",
    description="Phát hiện câu nói tiếng Việt đã hoàn chỉnh chưa (callbot turn detection)",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("%s %s %d  %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["info"])
def root():
    return {
        "service": "SBD Prediction API",
        "model":   MODEL_DIR,
        "threshold": THRESHOLD,
        "endpoints": {
            "POST /predict":       "single text",
            "POST /predict/batch": f"batch (max {MAX_BATCH})",
            "GET  /health":        "health check",
        },
    }


@app.get("/health", tags=["info"])
def health():
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":    "ok",
        "model_dir": MODEL_DIR,
        "threshold": predictor.threshold,
        "device":    predictor.device,
    }


@app.post("/predict", tags=["predict"])
def predict(req: PredictRequest):
    """
    Predict một câu.

    Request:  `{"text": "Tôi muốn đặt lịch khám ngày mai"}`

    Response:
    ```json
    {
      "text": "Tôi muốn đặt lịch khám ngày mai",
      "label": "complete",
      "is_complete": true,
      "prob_complete": 0.9712,
      "prob_incomplete": 0.0288,
      "latency": {
        "tokenize_ms": 1.2,
        "inference_ms": 7.8,
        "total_ms": 9.4
      }
    }
    ```
    """
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predictor.predict_one(req.text)
    except Exception as exc:
        logger.exception("Predict error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", tags=["predict"])
def predict_batch(req: BatchRequest):
    """
    Predict nhiều câu trong một request (hiệu quả hơn gọi nhiều lần).

    Request:  `{"texts": ["câu 1", "câu 2", ...]}`

    Response:
    ```json
    {
      "results": [
        {"text": "câu 1", "label": "complete", "is_complete": true, "prob_complete": 0.97, "prob_incomplete": 0.03},
        {"text": "câu 2", "label": "incomplete", "is_complete": false, "prob_complete": 0.03, "prob_incomplete": 0.97}
      ],
      "latency": {
        "tokenize_ms": 1.5,
        "inference_ms": 44.2,
        "total_ms": 46.8,
        "per_item_ms": 23.4
      }
    }
    ```
    """
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predictor.predict_batch(req.texts)
    except Exception as exc:
        logger.exception("Batch predict error")
        raise HTTPException(status_code=500, detail=str(exc))
