# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vietnamese Sentence Boundary Detection (SBD) for callbot turn detection — classifies whether a customer utterance is **complete** (label 1) or **incomplete** (label 0). Built on PhoBERT (`vinai/phobert-base`) fine-tuned on synthetic Vietnamese call-center data generated via OpenAI.

## Environment Setup

```bash
# For API server only
make install                   # creates .venv + installs requirements.txt

# For training + data generation
make install-train             # creates .venv + installs requirements-train.txt
```

Requires a `.env` file (see `.env.example`). `OPENAI_API_KEY` is required only for data generation.

## Common Commands

```bash
# API server
make serve                     # uvicorn on port 8000

# Full ML pipeline (in order)
make test-sample               # sanity-check 20 generated examples
make generate                  # generate ~3800 training examples (10–15 min)
make train                     # fine-tune PhoBERT (5 epochs)
make evaluate                  # evaluate on test set
make export                    # export to ONNX

# Inference
make inference                 # interactive CLI demo
.venv/bin/python scripts/inference.py --model-dir runs/sbd_model/final --text "Tôi muốn đặt lịch"

# Bash wrapper (calls the API)
./predict.sh "Tôi muốn đặt lịch"
THRESHOLD=0.70 ./predict.sh "câu 1" "câu 2"

# Run with custom training args
.venv/bin/python scripts/train.py --epochs 3 --lr 1e-5

# Evaluate a specific split
.venv/bin/python scripts/evaluate.py --model-dir runs/sbd_model/final --split val
```

## Architecture

### Data flow

```
OpenAI (gpt-4o-mini)
  └─ src/data_generation/generate.py   # generates JSONL examples per scenario
       └─ scripts/generate_data.py     # orchestrates full dataset split
            └─ data/processed/{train,val,test}.jsonl
                 └─ src/training/dataset.py (SBDDataset)
                      └─ scripts/train.py   # HuggingFace Trainer → runs/sbd_model/final/
                           └─ api/predictor.py (SBDPredictor singleton)
                                └─ api/server.py (FastAPI)
```

### Key design decisions

**`api/predictor.py`** — singleton loaded once at FastAPI startup via `init_predictor()`, accessed via `get_predictor()`. Each uvicorn worker loads its own copy. Threshold defaults to 0.65 (overridable via `SBD_THRESHOLD` env var — lower = cuts turn sooner, higher = waits longer).

**`src/training/dataset.py`** — PhoBERT is RoBERTa-based so it has no `token_type_ids`; the dataset conditionally includes them only if the tokenizer returns them (for other model options like multilingual BERT).

**`api/server.py`** — uses a custom `UTF8JSONResponse` to prevent Vietnamese text from being ASCII-escaped in JSON output. Batch endpoint returns results array with a `{"batch_latency_ms": ...}` dict appended as the last item.

**`config/config.yaml`** — single source of truth for all hyperparameters, data paths, and generation scenarios. Scripts read this file at runtime; CLI flags (`--epochs`, `--lr`) override config values when provided.

### Model alternatives (in config)

| Option | Model | Trade-off |
|--------|-------|-----------|
| A (default) | `vinai/phobert-base` | Best accuracy for Vietnamese |
| B | `distilbert-base-multilingual-cased` | Smaller, faster |
| C | `google-bert/bert-base-multilingual-cased` | Balanced |

### Data format (JSONL)

```json
{"text": "Tôi muốn đặt lịch khám ngày mai", "label": 1}
{"text": "Tôi muốn", "label": 0}
```

Labels: `0` = incomplete, `1` = complete. Dataset is ~50/50 balanced by construction (generation script enforces equal counts per scenario).

### Production deployment

```bash
sudo bash deploy.sh            # installs to /opt/sbd, creates systemd service
```

Service file: `sbd.service`. Env vars for production override: `SBD_MODEL_DIR`, `SBD_THRESHOLD`, `SBD_MAX_BATCH` (default 32), `LOG_LEVEL`.

### API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service info |
| GET | `/health` | Model status, device, threshold |
| POST | `/predict` | Single text → classification |
| POST | `/predict/batch` | Up to 32 texts → classifications |
