# 🎙️ Vietnamese Sentence Boundary Detection

Phát hiện ranh giới câu nói tiếng Việt cho hệ thống callbot — xác định khi nào người dùng **nói xong lượt** để bot phản hồi đúng lúc.

[![Model](https://img.shields.io/badge/model-PhoBERT--base-blue)](https://huggingface.co/vinai/phobert-base)
[![Accuracy](https://img.shields.io/badge/accuracy-95.05%25-brightgreen)]()
[![Latency](https://img.shields.io/badge/latency-%3C3ms-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

## Kết quả

| Metric | Giá trị |
|--------|---------|
| Accuracy (test_v2, 1940 mẫu) | **95.05%** |
| F1 macro | 0.939 |
| F1 incomplete (tránh ngắt lời) | 0.966 |
| F1 complete | 0.913 |
| False Positive (ngắt lời sai) | 2.5% |
| Inference latency | **< 3ms**/câu (batch, CPU) |

## Bài toán

Trong callbot, STT (Speech-to-Text) trả về text liên tục từng chunk. Cần phân loại:

```
"Tôi muốn"              → INCOMPLETE  (chờ người dùng nói tiếp)
"Tôi muốn đặt lịch khám" → COMPLETE   (bot có thể phản hồi)
"ờ anh thấy cũng ok đó"  → COMPLETE   (xác nhận, kết thúc lượt)
"ờ thì là"               → INCOMPLETE  (filler, chưa xong)
```

## Kiến trúc

```
ASR text
  └─ clean_asr()              # strip punctuation (ASR không có dấu câu)
       └─ PhoBERT-base        # fine-tuned binary classifier
            └─ post-process   # dangling word override + ACK word shortcut
                 └─ label: complete / incomplete
```

**PhoBERT-base** (vinai) được fine-tune thêm classifier head 2 lớp, trained trên **9.892 mẫu** tiếng Việt tổng hợp bao gồm:
- Call center inbound/outbound
- Khảo sát CSAT/NPS
- Telesale (bảo hiểm, ngân hàng, khóa học)
- Câu ngập ngừng, tự sửa, giọng Nam/Bắc

## Cài đặt

```bash
git clone https://github.com/TechGuyVN/Sentence-Boundary-Detection.git
cd Sentence-Boundary-Detection

# API server (production)
make install        # tạo .venv + cài requirements.txt
make serve          # uvicorn trên port 8000

# Training pipeline (cần thêm deps)
make install-train  # cài requirements-train.txt
cp .env.example .env && vi .env   # thêm OPENAI_API_KEY
```

## Sử dụng nhanh

### Bash script

```bash
./predict.sh "Tôi muốn đặt lịch khám ngày mai"
# → {"label": "COMPLETE", "is_complete": true, "prob_complete": 0.97, "latency_ms": 9.2}

./predict.sh "câu 1" "câu 2" "câu 3"    # nhiều câu, model load 1 lần
THRESHOLD=0.70 ./predict.sh "text..."   # custom threshold
```

### Python

```python
from scripts.inference import SBDPredictor

predictor = SBDPredictor("runs/sbd_model/final", threshold=0.65)

result = predictor.predict("Tôi muốn đặt lịch khám ngày mai")
# {
#   "label": "complete",
#   "is_complete": True,
#   "prob_complete": 0.9712,
#   "latency_ms": 8.4
# }
```

### API Server

```bash
make serve   # hoặc: uvicorn api.server:app --host 0.0.0.0 --port 8000
```

```bash
# Single
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tôi muốn đặt lịch khám"}'

# Batch (hiệu quả hơn)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["câu 1", "câu 2", "câu 3"]}'

# Health check
curl http://localhost:8000/health
```

**Response:**
```json
{
  "text": "Tôi muốn đặt lịch khám ngày mai",
  "label": "complete",
  "is_complete": true,
  "prob_complete": 0.9712,
  "prob_incomplete": 0.0288,
  "latency_ms": 8.4
}
```

## Deploy lên Linux server

```bash
# Trên server Ubuntu/Debian, chạy với root
sudo bash deploy.sh

# Quản lý service
systemctl status sbd
journalctl -u sbd -f          # xem log realtime

# Đổi threshold không cần restart code
systemctl edit sbd             # thêm Environment=SBD_THRESHOLD=0.70
systemctl restart sbd
```

## Training pipeline

```bash
# 1. Sinh data bằng OpenAI
make test-sample               # kiểm tra chất lượng 20 mẫu
make generate                  # sinh ~3800 mẫu (10-15 phút)

# 2. Train
make train                     # fine-tune PhoBERT (5 epochs, ~25 phút CPU)

# 3. Đánh giá
make evaluate                  # test set accuracy
python scripts/test_cases.py   # 231 test case thủ công

# 4. Export ONNX (optional, cho production)
make export
```

## Cấu trúc project

```
├── api/
│   ├── predictor.py     # SBDPredictor singleton, post-processing logic
│   └── server.py        # FastAPI app, /predict, /predict/batch, /health
├── config/
│   └── config.yaml      # model, data generation, training hyperparameters
├── scripts/
│   ├── train.py         # HuggingFace Trainer fine-tune
│   ├── evaluate.py      # evaluate on any split
│   ├── inference.py     # SBDPredictor + interactive CLI demo
│   ├── generate_data.py # sinh data tổng hợp bằng OpenAI
│   ├── merge_and_split.py     # dedup + balance + split data
│   ├── clean_punctuation.py   # strip dấu câu khỏi training data
│   └── test_cases.py    # 231 hand-labeled test cases
├── src/
│   ├── data_generation/ # OpenAI generation logic
│   ├── training/        # PyTorch Dataset
│   └── utils/           # metrics (F1, confusion matrix)
├── runs/sbd_model/final/  # trained model weights (Git LFS)
├── predict.sh           # CLI wrapper
├── deploy.sh            # Linux server deploy script
├── sbd.service          # systemd service file
└── requirements.txt     # API server deps (torch, transformers, fastapi)
```

## Hyperparameters

| Parameter | Giá trị |
|-----------|---------|
| Base model | `vinai/phobert-base` |
| Max sequence length | 128 tokens |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 5 |
| Warmup ratio | 0.1 |
| Decision threshold | 0.65 (tunable) |

## Threshold tuning

| Threshold | Hành vi |
|-----------|---------|
| 0.60 | Nhạy hơn — bot phản hồi nhanh hơn, có thể ngắt lời nhiều hơn |
| **0.65** | **Mặc định — cân bằng tốt** |
| 0.70 | Kiên nhẫn hơn — ít ngắt lời, bot chờ lâu hơn một chút |

Đổi threshold qua env var: `SBD_THRESHOLD=0.70 ./predict.sh "text..."`

## Lưu ý với ASR

Model được train và inference **không có dấu câu** (ASR thực tế không sinh `.`, `,`, `?`). Input từ STT engine sẽ được tự động strip punctuation trước khi inference.

## License

MIT
