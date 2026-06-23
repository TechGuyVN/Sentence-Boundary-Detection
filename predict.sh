#!/usr/bin/env bash
# Usage:
#   ./predict.sh "text..."                    # single text
#   ./predict.sh "câu 1" "câu 2" "câu 3"    # multiple texts (model loaded once)
#   echo "text" | ./predict.sh               # from stdin
#   THRESHOLD=0.70 ./predict.sh "text..."    # custom threshold

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/runs/sbd_model/final"
PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
THRESHOLD="${THRESHOLD:-0.65}"

# Read from stdin if no args
if [[ $# -eq 0 ]]; then
  if [[ -t 0 ]]; then
    echo "Usage: ./predict.sh \"text câu nói\"" >&2
    echo "       ./predict.sh \"câu 1\" \"câu 2\" ..." >&2
    echo "       echo \"text\" | ./predict.sh" >&2
    exit 1
  fi
  mapfile -t TEXTS
else
  TEXTS=("$@")
fi

# Build JSON array of texts to pass to Python
TEXTS_JSON=$(python3 -c "
import sys, json
texts = sys.argv[1:]
print(json.dumps(texts))
" "${TEXTS[@]}")

"$PYTHON" - "$MODEL_DIR" "$THRESHOLD" "$TEXTS_JSON" 2>/dev/null <<'PYEOF'
import sys, json, time, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir  = sys.argv[1]
threshold  = float(sys.argv[2])
texts      = json.loads(sys.argv[3])

# ASR không sinh dấu câu — strip trước khi inference
_STRIP_RE = re.compile(r"[.!?,;:\"'()\[\]{}\-–—…]")
def clean_asr(t):
    return re.sub(r"\s{2,}", " ", _STRIP_RE.sub(" ", t)).strip()
texts = [clean_asr(t) for t in texts]

_DANGLING_RE = re.compile(
    r"""(?ix)\b(cũng|thì|là|và|hoặc|nhưng|mà|để|vì|bởi|rằng|hay|nếu|khi|sẽ|bị|nên|vẫn|chỉ|tuy\s+nhiên|trong\s+vòng|bởi\s+vì|chính\s+là)\s*[.!?,;]?\s*$""",
    re.UNICODE,
)
_ACK_RE = re.compile(
    r"""(?ix)^\s*(dạ|vâng|ừ|ok|okay|rồi|thôi|xong|được|cũng\s+được|thôi\s+được|ừ\s+được|ừ\s+cũng\s+được|dạ\s+rồi|vâng\s+rồi|ok\s+rồi|thôi\s+thì\s+thôi|không\s+cần|không\s+sao|không\s+cần\s+đâu)\s*[.!,]?\s*$""",
    re.UNICODE,
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

results = []
for text in texts:
    t0 = time.perf_counter()
    enc = tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        ).logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    ms = round((time.perf_counter() - t0) * 1000, 1)
    p_complete = probs[1]

    overridden = None
    if _ACK_RE.match(text):
        p_complete = max(probs[1], threshold + 0.01)
        overridden = "ack_word"
    elif p_complete >= threshold and _DANGLING_RE.search(text.strip()):
        p_complete = min(probs[0], threshold - 0.01)
        overridden = "dangling_word"

    row = {
        "text":            text,
        "label":           "COMPLETE" if p_complete >= threshold else "INCOMPLETE",
        "is_complete":     p_complete >= threshold,
        "prob_complete":   round(p_complete, 4),
        "prob_incomplete": round(1 - p_complete, 4),
        "threshold":       threshold,
        "latency_ms":      ms,
    }
    if overridden:
        row["override"] = overridden
    results.append(row)

if len(results) == 1:
    print(json.dumps(results[0], ensure_ascii=False, indent=2))
else:
    print(json.dumps(results, ensure_ascii=False, indent=2))
PYEOF
