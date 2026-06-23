#!/usr/bin/env python3
"""
Sinh data nhắm vào pattern sai từ test_v2.jsonl.
Chạy xong tự merge + retrain.
"""
import json, os, sys, time, re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

SYSTEM = """Bạn tạo dữ liệu huấn luyện SBD cho callbot tiếng Việt.
label=1: người dùng NÓI XONG lượt nói, callbot có thể trả lời.
label=0: người dùng CHƯA XONG, đang setup context hoặc bỏ dở.
Trả về JSON: {"examples": [{"text": "...", "label": 0|1}]}"""

GROUPS = [
    # ══ FALSE POSITIVE fixes — phải là INCOMPLETE (label=0) ══════════════════
    (
        "framing_thinking",
        250,
        """Tạo câu tiếng Việt callbot dạng "ờ/à/ừ [ai] nghĩ là [chủ đề]" hoặc
"nói chung là [chủ đề]" hoặc "theo [ai] thì [chủ đề]" — người nói đang
SETUP ngữ cảnh, chưa hỏi xong → label=0 (INCOMPLETE).

Ví dụ INCOMPLETE (label=0):
- "ờ em nghĩ là bên mình có hỗ trợ token API"
- "nói chung là bên mình có hỗ trợ template ZNS"
- "theo anh thì cái dashboard cho sale này rồi"
- "theo em thì bên mình có hỗ trợ ghi âm cuộc gọi"
- "nói chung là anh muốn kiểm tra lại phần cấu hình"

Ví dụ COMPLETE (label=1) để phân biệt:
- "ờ em nghĩ là được rồi"
- "nói chung anh hài lòng với dịch vụ"
- "theo em thì gói này phù hợp"

Tạo {n} ví dụ, 65% INCOMPLETE. Đa dạng chủ đề: API, template ZNS, SIP trunk,
báo cáo realtime, hệ thống ticket, tích hợp CRM, gói cước, đầu số hotline.""",
        0.65,
    ),
    (
        "hypothetical_setup",
        200,
        """Tạo câu tiếng Việt callbot dạng "giả sử như X", "có vẻ là X",
"hình như là X", "chắc là X" — đang nêu điều kiện/phỏng đoán, chưa hỏi xong → label=0.

Ví dụ INCOMPLETE (label=0):
- "giả sử như bên mình có hỗ trợ báo cáo realtime"
- "có vẻ là bên em muốn tích hợp CRM"
- "hình như là bên mình có hỗ trợ tích hợp với website"
- "chắc là bên mình có hỗ trợ báo cáo realtime"
- "giả sử như anh muốn kiểm tra lại phần kịch bản"
- "có vẻ là anh muốn kiểm tra lại phần hệ thống ticket"

Ví dụ COMPLETE (label=1):
- "có vẻ dịch vụ ổn hơn trước rồi"
- "chắc tôi sẽ đăng ký gói này"
- "hình như là tôi hiểu rồi"

Tạo {n} ví dụ, 65% INCOMPLETE.""",
        0.65,
    ),
    (
        "context_setup_before_complaint",
        250,
        """Tạo câu tiếng Việt callbot dạng mô tả sự việc/tình huống trước khi
hỏi/khiếu nại — câu CHƯA XONG vì người dùng sắp nói thêm → label=0.

Ví dụ INCOMPLETE (label=0):
- "Hôm qua tôi có mua một cái máy mới"       ← sắp nói vấn đề
- "điện nước tháng này tôi thấy hơi cao"      ← sắp khiếu nại
- "mạng mình hay bị đứt lắm"                  ← sắp báo cáo sự cố
- "tôi không thấy ghi rõ số tiền trong hóa đơn"  ← sắp hỏi thêm
- "nhân viên nói chuyện thân thiện"           ← đang đánh giá, chưa xong
- "cái này thật sự làm tôi khó chịu"         ← sắp giải thích vấn đề

Ví dụ COMPLETE (label=1):
- "hôm qua mạng bị đứt, tôi muốn khiếu nại"
- "tôi phát hiện hóa đơn tháng này sai, cần hỗ trợ"
- "điện nước tháng này cao bất thường"        ← đây là phản hồi khảo sát, xong

Tạo {n} ví dụ, 60% INCOMPLETE. Đa dạng lĩnh vực: điện nước, mạng internet,
thiết bị điện tử, dịch vụ ngân hàng, bảo hiểm, khóa học.""",
        0.60,
    ),
    (
        "third_person_needs",
        200,
        """Tạo câu tiếng Việt callbot dạng "[người] muốn/cần/đang X" —
nhân viên/kế toán/sale... muốn gì đó nhưng chưa nói xong → label=0.

Ví dụ INCOMPLETE (label=0):
- "kế toán muốn thanh toán"
- "sale muốn tích hợp"
- "nhân viên đang gặp lỗi"
- "trưởng nhóm muốn xem báo cáo"
- "alo kế toán muốn biết tại sao"
- "sale muốn kích hoạt"
- "admin muốn xuất báo cáo theo"

Ví dụ COMPLETE (label=1):
- "kế toán muốn thanh toán hóa đơn tháng này"
- "sale cần tích hợp với phần mềm CRM Hubspot"
- "nhân viên bị lỗi đăng nhập, cần reset mật khẩu"

Tạo {n} ví dụ, 60% INCOMPLETE. Đa dạng vai trò: kế toán, sale, nhân viên,
trưởng nhóm, admin, đại lý, khách hàng, người dùng.""",
        0.60,
    ),
    (
        "de_xem_setup",
        150,
        """Tạo câu tiếng Việt callbot dạng "để anh/em xem X" hay "để anh/em xem X nói sao ta"
— đang mở đầu giải thích, chưa xong → label=0.

Ví dụ INCOMPLETE (label=0):
- "để anh xem cái ZNS chăm sóc khách hàng này nói sao ta"
- "để em xem anh muốn kiểm tra lại phần gói tổng đài ảo"
- "để anh xem khách bên em hỏi về luồng gọi tự động rồi"
- "để em xem anh muốn kiểm tra lại phần báo cáo cuộc gọi nhỡ"
- "để anh xem bên em muốn kịch bản chăm sóc khách hàng"

Ví dụ COMPLETE (label=1):
- "để anh xem lại đã nhé"
- "để em kiểm tra thông tin cho anh"
- "để anh xem và gọi lại sau"

Tạo {n} ví dụ, 65% INCOMPLETE. Đa dạng chủ đề kỹ thuật callcenter/SaaS.""",
        0.65,
    ),
    # ══ FALSE NEGATIVE fixes — phải là COMPLETE (label=1) ════════════════════
    (
        "em_hoi_chut_complete",
        250,
        """Tạo câu tiếng Việt dạng "em hỏi chút X" hoặc "mình hỏi chút X" —
đã nói ĐỦ thông tin, callbot có thể trả lời → label=1 (COMPLETE).

Ví dụ COMPLETE (label=1):
- "em hỏi chút anh muốn hỏi bên mình có hỗ trợ ESMS không"
- "em hỏi chút bên em muốn nâng cấp gói cước từ tháng sau"
- "mình hỏi chút bên mình có hỗ trợ ZNS không"
- "em hỏi chút em muốn đổi mật khẩu tài khoản admin"
- "em hỏi chút callbot có tích hợp được với CRM hiện tại không"
- "em hỏi chút khách hàng yêu cầu xuất file ghi âm để đối soát"
- "em hỏi chút bên anh cần tích hợp API gửi OTP qua SMS"

Ví dụ INCOMPLETE (label=0):
- "em hỏi chút là"
- "mình hỏi chút về"
- "em muốn hỏi thêm về"

Tạo {n} ví dụ, 70% COMPLETE. Đa dạng nội dung: tính năng sản phẩm, hỗ trợ
kỹ thuật, tài khoản, hóa đơn, tích hợp API, gói cước.""",
        0.70,
    ),
    (
        "short_direct_question_complete",
        250,
        """Tạo câu hỏi ngắn tiếng Việt callbot — đã hỏi ĐỦ Ý, không cần nói thêm → label=1.

Ví dụ COMPLETE (label=1):
- "Giá phòng bao nhiêu"
- "thẻ này có ưu đãi gì"
- "học phí khóa học này là bao nhiêu vậy"
- "thời hạn đăng ký là bao lâu"
- "có chương trình khuyến mãi không"
- "Tại sao lại như vậy"
- "mạng chậm quá"
- "hơi lâu một chút"
- "cái này không được đúng"
- "Dạ cho tôi biết thời gian"
- "Giá hơi cao"
- "đường truyền có vấn đề"

Ví dụ INCOMPLETE (label=0):
- "Giá phòng thì"
- "thẻ này có ưu đãi thì"
- "học phí thì"

Tạo {n} ví dụ, 70% COMPLETE (câu hỏi/nhận xét ngắn đầy đủ ý).
Đa dạng: câu hỏi giá/thời gian/điều kiện, nhận xét ngắn 2-5 từ,
phàn nàn ngắn, xác nhận thông tin.""",
        0.70,
    ),
    (
        "short_opinion_feedback_complete",
        200,
        """Tạo câu nhận xét/phản hồi ngắn tiếng Việt callbot — người nói đã nói XONG → label=1.

Ví dụ COMPLETE (label=1):
- "dạ cái đó tôi không biết"
- "anh thấy sản phẩm này khá hay"
- "ừm thì tôi đồng ý mà"
- "Khóa học này rất hay"
- "thật sự thất vọng về dịch vụ"
- "khóa học đẹp quá"
- "mạng chậm quá"
- "tôi không nhận được tin nhắn xác nhận"
- "Cả nhà ai cũng dùng cái này"
- "ờ thì sản phẩm này có tốt không"   ← câu hỏi = complete
- "được rồi tôi cần thêm thông tin"
- "Để anh hỏi thêm ý kiến đã"

Ví dụ INCOMPLETE (label=0):
- "tôi thấy sản phẩm này"
- "anh đồng ý nhưng"
- "ừm thì"

Tạo {n} ví dụ, 65% COMPLETE. Đa dạng: hài lòng/không hài lòng, ngắn/vừa,
giọng Bắc/Nam, bối cảnh callcenter/telesale/khảo sát.""",
        0.65,
    ),
]


def gen_group(client, name, n, prompt_tmpl):
    prompt = prompt_tmpl.replace("{n}", str(n))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.95,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    examples = data.get("examples", data) if isinstance(data, dict) else data

    _PUNCT = re.compile(r"[.!?,;:\"'()\[\]{}\-–—…]")
    result = []
    for e in examples:
        text  = e.get("text") or e.get("utterance") or e.get("sentence") or ""
        label = e.get("label", -1)
        if text and label in (0, 1, "0", "1"):
            clean = re.sub(r"\s{2,}", " ", _PUNCT.sub(" ", str(text))).strip()
            result.append({"text": clean, "label": int(label)})
    return result


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out_path = Path("data/raw/from_errors.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_new = []
    for name, n, prompt, ratio in tqdm(GROUPS, desc="Groups"):
        batch_size, group = 30, []
        for _ in range((n + batch_size - 1) // batch_size):
            if len(group) >= n: break
            try:
                group.extend(gen_group(client, name, min(batch_size, n - len(group)), prompt))
                time.sleep(0.3)
            except Exception as ex:
                print(f"\n[WARN] {name}: {ex}")
                time.sleep(2)
        print(f"  {name}: {len(group)} mẫu")
        all_new.extend(group)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_new:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(e["label"] for e in all_new)
    print(f"\nSaved {len(all_new)} → {out_path}")
    print(f"  complete={c[1]} ({100*c[1]/len(all_new):.1f}%)  incomplete={c[0]}")
    print("\nTiếp theo: python scripts/merge_and_split.py --balance && python scripts/train.py")


if __name__ == "__main__":
    main()
