"""
Generate synthetic Vietnamese SBD training data via OpenAI.

Each example is:
  {"text": "...", "label": 0|1}
  label 0 = incomplete utterance
  label 1 = complete utterance (speaker finished turn)
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = """Bạn là chuyên gia tạo dữ liệu huấn luyện cho bài toán
Sentence Boundary Detection (SBD) trong hệ thống callbot tiếng Việt.

Nhiệm vụ: Tạo các câu mà người dùng nói với callbot, gán nhãn hoàn chỉnh / chưa hoàn chỉnh.

Định nghĩa:
- complete (label=1): Người nói đã nói XONG Ý, callbot có thể phản hồi.
- incomplete (label=0): Người nói chưa xong, đang ngập ngừng hoặc dừng giữa câu.

═══ QUY TẮC PHÂN LOẠI CHI TIẾT ═══

[COMPLETE — label=1]:
• Câu kết thúc bằng dấu câu: ".", "?", "!"
• Câu KHÔNG có dấu câu nhưng ý đầy đủ: "tôi muốn hủy đơn hàng đó", "anh gửi thông tin qua zalo cho em"
• Câu đồng ý / phủ nhận ngắn gọn: "được", "ok", "vâng", "không cần đâu", "thôi thì thôi"
• Câu xã giao kết thúc hội thoại: "cảm ơn em", "ok cảm ơn nhé", "vâng cho anh hỏi thêm sau"
• Câu phản hồi khảo sát / đánh giá: "anh cho 8 điểm", "tôi thấy ổn", "dịch vụ khá tốt"
• ĐẶC BIỆT — câu bắt đầu bằng "ờ/ừ/à" nhưng KẾT THÚC có ý hoàn chỉnh là COMPLETE:
  - "ờ anh thấy cũng ok đó em à" → COMPLETE
  - "ừ thì tôi đồng ý với mức đó" → COMPLETE
  - "à vâng anh hiểu rồi" → COMPLETE
  - "ờ thì thôi được rồi" → COMPLETE

[INCOMPLETE — label=0]:
• Dừng giữa câu, câu chưa có vị ngữ: "tôi muốn hỏi về", "cái đó thì"
• Chỉ là filler rồi dừng, KHÔNG có thêm nội dung: "ờ thì là", "ừm", "à thì"
• Liệt kê chưa xong: "tên tôi là Nguyễn Văn", "số điện thoại của tôi là"
• Câu bị cắt giữa chừng rõ ràng: "tôi cần", "cho tôi hỏi là"

═══ QUY TẮC TẠO DỮ LIỆU ═══
1. KHÔNG thêm dấu câu vào câu incomplete.
2. Đa dạng độ dài: 2-4 từ (ngắn), 5-12 từ (trung bình), 13-30 từ (dài).
3. Bao gồm: số điện thoại, tên người, mã đơn hàng, ngày giờ, địa chỉ.
4. Giọng vùng miền: dùng "dạ/ạ" (miền Nam), "vâng/ạ" (miền Bắc), "mô/chi" (miền Trung).
5. Đặc thù kịch bản: call center dùng từ chuyên nghiệp hơn; telesale/khảo sát dùng ngôn ngữ tự nhiên hơn."""

USER_PROMPT_TEMPLATE = """Kịch bản: {scenario}

Tạo đúng {n} ví dụ, cân bằng 50% complete (label=1) và 50% incomplete (label=0).
Yêu cầu bắt buộc:
- Độ dài đa dạng: ~30% ngắn (2-5 từ), ~40% trung bình (6-15 từ), ~30% dài (16-30 từ)
- Ngôn ngữ tự nhiên như lời nói thực, KHÔNG văn chương
- Bao gồm câu bắt đầu "ờ/ừ/à" nhưng KẾT THÚC có ý => label=1 (xem quy tắc)
- Đặc thù kịch bản: từ chuyên ngành, phản ứng cảm xúc, từ chối/đồng ý

Trả về JSON array hợp lệ."""


def generate_batch(
    client: OpenAI,
    scenario: str,
    n: int,
    model: str = "gpt-4o-mini",
    temperature: float = 0.9,
) -> list[dict]:
    """Call OpenAI and return a list of labeled examples for one scenario."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(scenario=scenario, n=n)
                + '\n\nTrả về {"examples": [...]}',
            },
        ],
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    # Support both {"examples": [...]} and bare [...]
    examples = data.get("examples", data) if isinstance(data, dict) else data
    result = []
    for e in examples:
        # Accept "text", "utterance", "sentence", "câu", "noi_dung"
        text = (e.get("text") or e.get("utterance") or e.get("sentence")
                or e.get("câu") or e.get("noi_dung") or "")
        label = e.get("label", e.get("nhãn", e.get("nhan", -1)))
        if text and label in (0, 1, "0", "1"):
            result.append({"text": str(text), "label": int(label)})
    return result


def generate_dataset(
    scenarios: list[str],
    total_samples: int,
    batch_size: int = 20,
    model: str = "gpt-4o-mini",
    temperature: float = 0.9,
) -> Iterator[dict]:
    """Yield examples across all scenarios until total_samples is reached."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    samples_per_scenario = max(batch_size, total_samples // len(scenarios))
    generated = 0

    with tqdm(total=total_samples, desc="Generating examples") as pbar:
        for scenario in scenarios:
            remaining = total_samples - generated
            if remaining <= 0:
                break
            n_this = min(samples_per_scenario, remaining)
            batches_needed = (n_this + batch_size - 1) // batch_size

            for _ in range(batches_needed):
                remaining_now = total_samples - generated
                if remaining_now <= 0:
                    break
                n_batch = min(batch_size, remaining_now)
                try:
                    batch = generate_batch(client, scenario, n_batch, model, temperature)
                    for ex in batch:
                        yield ex
                        generated += 1
                        pbar.update(1)
                        if generated >= total_samples:
                            return
                    time.sleep(0.3)  # rate-limit courtesy
                except Exception as exc:
                    print(f"\n[WARN] Batch failed for '{scenario}': {exc}")
                    time.sleep(2)


def save_jsonl(examples: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


def split_and_save(
    all_examples: list[dict],
    train_path: str,
    val_path: str,
    test_path: str,
    val_ratio: float = 0.12,
    test_ratio: float = 0.08,
    seed: int = 42,
) -> None:
    random.seed(seed)
    random.shuffle(all_examples)
    n = len(all_examples)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = all_examples[:n_test]
    val = all_examples[n_test: n_test + n_val]
    train = all_examples[n_test + n_val:]
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    save_jsonl(test, test_path)
    print(f"\nSplit: train={len(train)} | val={len(val)} | test={len(test)}")
    _print_label_dist("train", train)
    _print_label_dist("val", val)
    _print_label_dist("test", test)


def _print_label_dist(name: str, examples: list[dict]) -> None:
    total = len(examples)
    n_complete = sum(1 for e in examples if e["label"] == 1)
    print(f"  {name}: complete={n_complete} ({100*n_complete/total:.1f}%)  incomplete={total-n_complete}")
