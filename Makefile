PYTHON=.venv/bin/python3
PIP=.venv/bin/pip

.PHONY: install test-sample generate train evaluate inference export clean

install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

install-train:
	python3 -m venv .venv
	$(PIP) install -r requirements-train.txt

# ── Start API server locally ────────────────────────────────────────────────
serve:
	$(PYTHON) -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1

# ── Step 1: sanity-check data quality (20 examples) ────────────────────────
test-sample:
	$(PYTHON) scripts/test_data_sample.py

# ── Step 2: generate full dataset (~3800 examples, 10-15 min) ──────────────
generate:
	$(PYTHON) scripts/generate_data.py

# ── Step 3: train ──────────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/train.py

# ── Step 4: evaluate on test set ──────────────────────────────────────────
evaluate:
	$(PYTHON) scripts/evaluate.py --model-dir runs/sbd_model/final

# ── Interactive inference demo ──────────────────────────────────────────────
inference:
	$(PYTHON) scripts/inference.py --model-dir runs/sbd_model/final

# ── Export to ONNX for production ──────────────────────────────────────────
export:
	$(PYTHON) scripts/export_onnx.py --model-dir runs/sbd_model/final

# ── Full pipeline ──────────────────────────────────────────────────────────
all: generate train evaluate export

clean:
	rm -rf runs/ data/processed/ exports/ __pycache__ src/**/__pycache__
