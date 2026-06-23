#!/usr/bin/env python3
"""
Sinh data nhắm vào các pattern model đang sai.
Chạy xong merge tự động vào processed/ rồi retrain.

Usage:
    python scripts/generate_targeted.py
"""
import json, os, sys, random, time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 7 nhóm pattern cần bổ sung ───────────────────────────────────────────────
# Mỗi nhóm: (tên, n_samples, prompt_mô_tả, tỉ_lệ_complete)
TARGETED_GROUPS = [
    (
        "la_complete",
        200,
        """Tạo câu tiếng Việt dạng "X là [giá trị đầy đủ]" — người dùng đã cung cấp thông tin xong.
Ví dụ COMPLETE:
- "Tên tôi là Nguyễn Văn An."
- "Địa chỉ của tôi là 123 Lê Lợi, quận 1."
- "Số hợp đồng là BH2024-00123."
- "Mã đơn hàng là DH-987654."
- "Ngày sinh của tôi là 15 tháng 3 năm 1990."
Ví dụ INCOMPLETE (câu bị cắt, chưa có giá trị):
- "Tên tôi là"
- "Địa chỉ là"
- "Mã số là"
Tạo {n} ví dụ, 60% COMPLETE, 40% INCOMPLETE. Đa dạng lĩnh vực: tên, địa chỉ, số hợp đồng, CMND, biển số xe, email.""",
        0.60,
    ),
    (
        "filler_action_complete",
        200,
        """Tạo câu tiếng Việt callbot: người dùng nói "ừ/ờ/à/dạ + hành động/ý kiến ngắn + vậy/thôi/đi/nhé/rồi".
Những câu này là COMPLETE dù bắt đầu bằng filler.
Ví dụ COMPLETE:
- "ừ thôi anh gọi lại sau vậy"
- "ờ cái đó tôi không cần"
- "à dạ thôi anh hủy đi"
- "ừ thôi để vậy đi em"
- "ờ anh không muốn nữa rồi"
- "dạ thôi em cứ xử lý đi"
- "à ừ anh chờ thêm chút nhé"
Ví dụ INCOMPLETE (filler rồi dừng):
- "ừ thì là"
- "ờ thì"
- "à mà tôi muốn"
Tạo {n} ví dụ, 65% COMPLETE, 35% INCOMPLETE. Đa dạng giọng Bắc/Nam, ngữ cảnh call center, telesale.""",
        0.65,
    ),
    (
        "de_postpone_complete",
        150,
        """Tạo câu tiếng Việt dạng "Để [ai] [làm gì] đã" hoặc "Để [ai] [làm gì] đã rồi tính/xem/hẵng" — người dùng hoãn lại, câu HOÀN CHỈNH.
Ví dụ COMPLETE:
- "Để anh nghĩ thêm đã."
- "Để tôi hỏi lại vợ đã rồi tính."
- "Để em kiểm tra lại đã nhé."
- "Để anh xem lịch đã."
- "Để tôi hỏi sếp đã rồi báo em."
- "Thôi để anh suy nghĩ thêm vậy."
- "Cho anh nghĩ thêm chút."
Ví dụ INCOMPLETE:
- "Để anh nghĩ thêm về"
- "Cho tôi hỏi thêm về vấn đề"
Tạo {n} ví dụ, 65% COMPLETE, 35% INCOMPLETE. Đa dạng chủ thể (anh/em/tôi/chị), hành động (nghĩ, hỏi, kiểm tra, xem xét).""",
        0.65,
    ),
    (
        "nhung_middle_complete",
        200,
        """Tạo câu tiếng Việt callbot có chứa "nhưng" GIỮA câu — câu vẫn HOÀN CHỈNH vì có mệnh đề phía sau "nhưng".
Ví dụ COMPLETE (nhưng ở giữa, có ý phía sau):
- "Dịch vụ ok nhưng giá hơi cao."
- "Nhân viên nhiệt tình nhưng chờ lâu quá."
- "Tôi hài lòng nhưng muốn hoàn tiền."
- "Gói này tốt nhưng tôi không cần."
- "Mạng ổn nhưng hay bị ngắt vào ban đêm."
- "Sản phẩm đẹp nhưng giao hàng chậm."
Ví dụ INCOMPLETE (nhưng ở CUỐI, chưa có ý):
- "Dịch vụ tốt nhưng"
- "Tôi muốn mua nhưng"
- "Sản phẩm ok nhưng"
Tạo {n} ví dụ, 55% COMPLETE (nhưng ở giữa), 45% INCOMPLETE (nhưng ở cuối).""",
        0.55,
    ),
    (
        "thi_ra_realization",
        150,
        """Tạo câu tiếng Việt dạng nhận ra/phát hiện: "Thì ra...", "À ra là...", "Ồ ra là...", "Thì ra là vậy", v.v. — câu HOÀN CHỈNH.
Ví dụ COMPLETE:
- "Thì ra là vậy."
- "À ra là thế."
- "Ồ thì ra vậy à."
- "À ra là anh không nhận được thông báo."
- "Thì ra hệ thống bị lỗi từ hôm qua."
- "À ra em gửi nhầm địa chỉ rồi."
- "Ồ thì ra phí đó là phí dịch vụ."
Ví dụ INCOMPLETE:
- "Thì ra là"
- "À ra thì"
- "Ồ vậy thì"
Tạo {n} ví dụ, 65% COMPLETE, 35% INCOMPLETE.""",
        0.65,
    ),
    (
        "short_negation_complete",
        150,
        """Tạo câu tiếng Việt callbot ngắn gọn: từ chối, phủ nhận, hoặc xác nhận không cần — câu HOÀN CHỈNH.
Ví dụ COMPLETE:
- "Không cần."
- "Thôi không cần đâu."
- "Anh không muốn."
- "Không, cảm ơn."
- "Tôi không quan tâm."
- "Thôi em ơi."
- "Không cần thiết."
- "Anh từ chối."
Ví dụ INCOMPLETE:
- "Tôi không"
- "Anh không muốn"
- "Không cần vì"
Tạo {n} ví dụ, 60% COMPLETE, 40% INCOMPLETE. Đa dạng từ chối lịch sự, thẳng thắn, có giải thích.""",
        0.60,
    ),
    (
        "vay_sentence_final",
        150,
        """Tạo câu tiếng Việt callbot kết thúc bằng "vậy", "vậy nhé", "vậy thôi", "vậy đi" — tiểu từ kết câu, câu HOÀN CHỈNH.
Ví dụ COMPLETE:
- "Thôi anh gọi lại sau vậy."
- "Tôi đặt lịch ngày mai vậy nhé."
- "Vậy thôi anh không mua nữa."
- "Thôi để vậy đi."
- "Anh cần suy nghĩ thêm vậy."
- "Ừ thôi vậy nhé."
Ví dụ INCOMPLETE (vậy ở giữa câu):
- "Nếu vậy thì"
- "Vậy thì anh cần"
- "Vậy mà tôi vẫn"
Tạo {n} ví dụ, 60% COMPLETE, 40% INCOMPLETE.""",
        0.60,
    ),
]

SYSTEM_PROMPT = """Bạn tạo dữ liệu huấn luyện SBD (Sentence Boundary Detection) cho callbot tiếng Việt.
- label=1: câu HOÀN CHỈNH, người nói đã nói xong, callbot có thể phản hồi.
- label=0: câu CHƯA XONG, đang bỏ dở hoặc cần tiếp tục.
Trả về JSON: {"examples": [{"text": "...", "label": 0 hoặc 1}, ...]}"""


def gen_group(client, name, n, prompt_tmpl, target_complete_ratio):
    prompt = prompt_tmpl.replace("{n}", str(n))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.95,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    data = json.loads(response.choices[0].message.content)
    examples = data.get("examples", data) if isinstance(data, dict) else data
    result = []
    for e in examples:
        text  = e.get("text") or e.get("utterance") or e.get("sentence") or ""
        label = e.get("label", -1)
        if text and label in (0, 1, "0", "1"):
            result.append({"text": str(text).strip(), "label": int(label)})
    return result


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base   = Path(__file__).parent.parent
    out_dir = base / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_new = []
    for name, n, prompt, ratio in tqdm(TARGETED_GROUPS, desc="Groups"):
        # Gọi theo batch nhỏ để tránh vượt token limit
        batch_size = 30
        group_examples = []
        calls = (n + batch_size - 1) // batch_size
        for _ in range(calls):
            try:
                batch = gen_group(client, name, min(batch_size, n - len(group_examples)), prompt, ratio)
                group_examples.extend(batch)
                time.sleep(0.3)
            except Exception as ex:
                print(f"\n[WARN] {name}: {ex}")
                time.sleep(2)
            if len(group_examples) >= n:
                break
        print(f"  {name}: {len(group_examples)} mẫu")
        all_new.extend(group_examples)

    # Lưu ra file riêng
    out_path = out_dir / "targeted.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_new:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(e["label"] for e in all_new)
    print(f"\nSaved {len(all_new)} targeted examples -> {out_path}")
    print(f"  complete={c[1]} ({100*c[1]/len(all_new):.1f}%)  incomplete={c[0]}")
    print("\nChạy tiếp: python scripts/merge_and_split.py --balance && python scripts/train.py")


if __name__ == "__main__":
    main()
