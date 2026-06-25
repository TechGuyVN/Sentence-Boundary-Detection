#!/usr/bin/env python3
"""
Gen thêm data cho ASR English phonetic cases.
70% label=1 (short confirmations / English loanwords = COMPLETE)
30% label=0 (same pattern nhưng chưa xong = INCOMPLETE)
"""
import json, os, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

SYSTEM = """Bạn tạo dữ liệu SBD cho callbot tiếng Việt với đặc thù ASR phiên âm tiếng Anh.
label=1: người nói ĐÃ XONG lượt nói — dù dùng tiếng Anh hay phiên âm.
label=0: người nói CHƯA XONG — câu còn dang dở.
Trả về JSON: {"examples": [{"text": "...", "label": 0|1}]}"""

GROUPS = [
    (
        "ok_variants_complete",
        150,
        """Tạo câu tiếng Việt callbot: người dùng xác nhận ngắn bằng "ok/oke/okay/ô kê" kết hợp
các từ khác nhau → COMPLETE (label=1).

Ví dụ COMPLETE:
- "ok em ghi nhận rồi" / "ô kê vậy anh" / "okay bên em hiểu rồi"
- "ok em xử lý liền" / "ô kê để em làm ngay" / "okay em sẽ liên hệ lại"
- "ok cảm ơn chị" / "ô kê cảm ơn anh" / "okay cảm ơn bạn"
- "ok rồi em xem lại" / "ô kê em sẽ gọi lại" / "okay xong rồi nhé"

Ví dụ INCOMPLETE (label=0):
- "ok nếu mà" / "ô kê bên em muốn" / "okay để em hỏi thêm về"

Tạo {n} ví dụ, 70% COMPLETE. Đa dạng: xác nhận, hứa hẹn xử lý, cảm ơn,
kết thúc cuộc gọi. Người nói có thể là nhân viên hoặc khách hàng."""
    ),
    (
        "yes_yeah_complete",
        120,
        """Tạo câu tiếng Việt callbot dùng "yes/yeah/yep/dét" — xác nhận HOÀN CHỈNH (label=1).

Ví dụ COMPLETE:
- "yes bên em nhận được rồi" / "yeah anh hiểu rồi" / "yep chị đúng"
- "yes đúng rồi em" / "yeah bên mình sẽ xử lý" / "yep ok rồi"
- "dét anh đồng ý" / "yes em ghi nhận" / "yeah cảm ơn chị"
- "ừ yes đúng vậy" / "à yeah ok rồi" / "yeah yep rồi"

Ví dụ INCOMPLETE (label=0):
- "yes nhưng mà" / "yeah nếu mà" / "yep để em xem là"

Tạo {n} ví dụ, 70% COMPLETE."""
    ),
    (
        "action_done_complete",
        150,
        """Tạo câu tiếng Việt callcenter: nhân viên báo đã HOÀN THÀNH hành động
(confirm/check/cancel/update/done/finish) → COMPLETE (label=1).

Ví dụ COMPLETE:
- "em confirm với anh rồi" / "bên em vừa confirm xong" / "em đã confirm thông tin"
- "em check lại rồi không có lỗi" / "anh check rồi thấy ổn"
- "em cancel đơn rồi chị" / "đã cancel thành công rồi"
- "update xong rồi em" / "bên em đã update thông tin"
- "done rồi anh nhé" / "xong hết rồi chị" / "hoàn tất rồi em"
- "em đã fix lỗi đó rồi" / "bug này fixed rồi anh"

Ví dụ INCOMPLETE (label=0):
- "confirm là" / "check lại phần" / "cancel thì cần" / "update thêm về"

Tạo {n} ví dụ, 70% COMPLETE. Đa dạng: vai trò nhân viên/khách, lĩnh vực
tổng đài/CNTT/ngân hàng/TMĐT."""
    ),
    (
        "ticket_issue_complete",
        120,
        """Tạo câu tiếng Việt callcenter: báo đã xử lý ticket/case/issue → COMPLETE.

Ví dụ COMPLETE:
- "em tạo ticket cho anh rồi" / "ticket mã TK-123 đã tạo"
- "case này em xử lý xong rồi" / "bên em đã close case đó"
- "issue đã resolve rồi anh" / "lỗi này em đã fix xong"
- "em đã assign ticket cho team kỹ thuật"
- "support team đã nhận case rồi chị"
- "em log ticket rồi sẽ có người liên hệ lại"

Ví dụ INCOMPLETE (label=0):
- "ticket nếu không" / "case này bên em muốn" / "issue thì cần"

Tạo {n} ví dụ, 70% COMPLETE."""
    ),
    (
        "payment_api_complete",
        120,
        """Tạo câu tiếng Việt: xác nhận payment/API/webhook/token thành công → COMPLETE.

Ví dụ COMPLETE:
- "payment thành công rồi anh" / "em thanh toán xong rồi"
- "API đang chạy ổn rồi" / "a p i kết nối được rồi" / "ây pi ai hoạt động rồi"
- "webhook nhận được data rồi" / "em test webhook thành công"
- "token còn hạn rồi anh" / "em lấy được token rồi"
- "callback trả về 200 rồi" / "CRM đã sync dữ liệu rồi"
- "integration chạy được rồi chị"

Ví dụ INCOMPLETE (label=0):
- "payment nếu bị lỗi thì" / "API bên em đang" / "webhook chưa"

Tạo {n} ví dụ, 70% COMPLETE."""
    ),
    (
        "phonetic_loanword_complete",
        120,
        """Tạo câu tiếng Việt với từ tiếng Anh bị ASR phiên âm sai → người dùng thực tế
đã nói xong (COMPLETE, label=1).

Các từ bị phiên âm thường gặp:
- cancel → căn sen / căn xồ / cần sồ
- confirm → con phơm
- check → chéc
- update → úp đết
- submit → sắp mít
- verify → vê ri phai
- done → đơn / đan
- finish → phí nít / phí ních
- complete → com pờ lít

Ví dụ COMPLETE:
- "căn sen rồi anh nhé" / "bên em đã căn xồ đơn"
- "con phơm thông tin rồi chị" / "đã con phơm hết rồi"
- "chéc xong rồi không có lỗi" / "anh chéc rồi thấy ổn"
- "úp đết thành công rồi em" / "đã úp đết xong"
- "phí nít rồi chị" / "đan rồi anh"

Ví dụ INCOMPLETE (label=0):
- "căn sen cái" / "con phơm thêm về" / "chéc giúp em"

Tạo {n} ví dụ, 70% COMPLETE. KHÔNG thêm dấu câu."""
    ),
    (
        "short_confirmation_complete",
        150,
        """Tạo câu xác nhận NGẮN tiếng Việt callbot — đủ ý, không cần nói thêm → COMPLETE.

Ví dụ COMPLETE:
- "dạ em hiểu" / "dạ anh nắm rồi" / "dạ chị biết rồi"
- "đúng rồi em" / "chính xác anh" / "chuẩn rồi chị"
- "vâng em ghi nhận" / "được anh" / "xong chị"
- "rồi em" / "rồi anh" / "ok rồi" / "đã rồi"
- "em nắm ý anh rồi" / "anh hiểu ý em rồi"
- "nhận được rồi chị" / "em tiếp nhận rồi"
- "anh đồng ý" / "chị đồng ý" / "em đồng ý"

Ví dụ INCOMPLETE (label=0):
- "dạ nếu mà" / "đúng nhưng" / "vâng thì" / "được thì"

Tạo {n} ví dụ, 70% COMPLETE. NHIỀU ĐA DẠNG, không lặp."""
    ),
]

_STRIP = re.compile(r"[.!?,;:\"'()\[\]{}\-–—…]")
def clean(t): return re.sub(r"\s{2,}", " ", _STRIP.sub(" ", str(t))).strip()

def gen_group(client, prompt, n):
    resp = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.95,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user",   "content": prompt.replace("{n}", str(n))}],
    )
    data = json.loads(resp.choices[0].message.content)
    examples = data.get("examples", data) if isinstance(data, dict) else data
    result = []
    for e in examples:
        text  = e.get("text") or e.get("utterance") or ""
        label = e.get("label", -1)
        if text and label in (0, 1, "0", "1"):
            result.append({"text": clean(text), "label": int(label)})
    return result

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out = Path("data/raw/asr_english_generated.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    all_new = []
    for name, n, prompt in tqdm(GROUPS, desc="Groups"):
        group, batch_size = [], 30
        for _ in range((n + batch_size - 1) // batch_size):
            if len(group) >= n: break
            try:
                group.extend(gen_group(client, prompt, min(batch_size, n - len(group))))
                time.sleep(0.3)
            except Exception as ex:
                print(f"\n[WARN] {name}: {ex}"); time.sleep(2)
        print(f"  {name}: {len(group)}")
        all_new.extend(group)

    with open(out, "w", encoding="utf-8") as f:
        for ex in all_new:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(e["label"] for e in all_new)
    print(f"\nSaved {len(all_new)} → {out}")
    print(f"  label=1: {c[1]} ({100*c[1]/len(all_new):.1f}%)  label=0: {c[0]}")

if __name__ == "__main__":
    main()
