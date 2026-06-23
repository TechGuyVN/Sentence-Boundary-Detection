#!/usr/bin/env python3
"""
Bộ test case thủ công — đánh giá độ chính xác trên các pattern thực tế callbot.
Mỗi case được gán nhãn bởi người, độc lập với tập train/val/test sinh bởi OpenAI.

Usage:
    python scripts/test_cases.py
    python scripts/test_cases.py --model-dir runs/sbd_model/final --threshold 0.65
    python scripts/test_cases.py --show-errors   # chỉ hiện case sai
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# ASR không sinh dấu câu — strip trước khi inference và trước khi hiển thị
_PUNCT_RE = re.compile(r"[.!?,;:\"'()\[\]{}\-–—…]")

def clean_asr(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()

# ── TEST CASES ────────────────────────────────────────────────────────────────
# Format: (text, expected_label, category)
# expected_label: 1=complete, 0=incomplete

TEST_CASES = [
    # ══════════════════════════════════════════════════════════════
    # 1. CÂU HOÀN CHỈNH — cơ bản
    # ══════════════════════════════════════════════════════════════
    ("Tôi muốn đặt lịch khám bệnh.",                        1, "basic_complete"),
    ("Cho tôi hỏi phí dịch vụ là bao nhiêu?",               1, "basic_complete"),
    ("Vâng, cảm ơn anh.",                                   1, "basic_complete"),
    ("Không, tôi không cần.",                               1, "basic_complete"),
    ("Được rồi.",                                           1, "basic_complete"),
    ("Ok em.",                                              1, "basic_complete"),
    ("Dạ vâng.",                                            1, "basic_complete"),
    ("Thôi được.",                                          1, "basic_complete"),
    ("Tôi đồng ý.",                                         1, "basic_complete"),
    ("Số điện thoại của tôi là 0912 345 678.",              1, "basic_complete"),
    ("Tên tôi là Nguyễn Văn An.",                           1, "basic_complete"),
    ("Tôi muốn hủy đơn hàng đó.",                           1, "basic_complete"),
    ("Gọi lại cho tôi vào buổi chiều nhé.",                 1, "basic_complete"),
    ("Tôi không hài lòng với dịch vụ này.",                 1, "basic_complete"),
    ("Cho tôi nói chuyện với quản lý.",                     1, "basic_complete"),

    # ══════════════════════════════════════════════════════════════
    # 2. CÂU CHƯA HOÀN CHỈNH — cơ bản
    # ══════════════════════════════════════════════════════════════
    ("Tôi muốn",                                            0, "basic_incomplete"),
    ("Cho tôi hỏi về",                                      0, "basic_incomplete"),
    ("Vâng thì",                                            0, "basic_incomplete"),
    ("Ờ thì là",                                            0, "basic_incomplete"),
    ("Số điện thoại của tôi là",                            0, "basic_incomplete"),
    ("Tên tôi là",                                          0, "basic_incomplete"),
    ("Tôi cần",                                             0, "basic_incomplete"),
    ("À thì",                                               0, "basic_incomplete"),
    ("Ừm",                                                  0, "basic_incomplete"),
    ("Ờ",                                                   0, "basic_incomplete"),
    ("Dạ thì",                                              0, "basic_incomplete"),
    ("Tôi muốn hỏi về vấn đề",                              0, "basic_incomplete"),
    ("Cái đó thì",                                          0, "basic_incomplete"),
    ("Anh ấy nói rằng",                                     0, "basic_incomplete"),

    # ══════════════════════════════════════════════════════════════
    # 3. DANGLING WORDS — từ cuối yêu cầu bổ ngữ
    # ══════════════════════════════════════════════════════════════
    ("ờ anh thấy cũng.",                                    0, "dangling_word"),
    ("tên tôi là.",                                         0, "dangling_word"),
    ("dịch vụ và.",                                         0, "dangling_word"),
    ("tôi muốn nhưng.",                                     0, "dangling_word"),
    ("anh thấy cũng",                                       0, "dangling_word"),
    ("dịch vụ cũng",                                        0, "dangling_word"),
    ("tôi nghĩ là",                                         0, "dangling_word"),
    ("bởi vì",                                              0, "dangling_word"),
    ("nếu mà",                                              0, "dangling_word"),
    ("tôi sẽ",                                              0, "dangling_word"),
    ("anh bị",                                              0, "dangling_word"),
    ("tôi cũng muốn",                                       0, "dangling_word"),
    ("chị ấy thì",                                          0, "dangling_word"),
    ("hay là",                                              0, "dangling_word"),
    ("tôi chỉ",                                             0, "dangling_word"),

    # ══════════════════════════════════════════════════════════════
    # 4. ỜỪÀ + Ý HOÀN CHỈNH — phải là COMPLETE
    # ══════════════════════════════════════════════════════════════
    ("ờ anh thấy cũng ok đó em à.",                         1, "filler_complete"),
    ("ừ thì tôi đồng ý với mức giá đó",                    1, "filler_complete"),
    ("à vâng anh hiểu rồi",                                 1, "filler_complete"),
    ("ờ thì thôi được rồi",                                 1, "filler_complete"),
    ("ừ được em ơi",                                        1, "filler_complete"),
    ("à không cần đâu",                                     1, "filler_complete"),
    ("ờ anh biết rồi cảm ơn",                               1, "filler_complete"),
    ("ừ thôi anh gọi lại sau vậy",                          1, "filler_complete"),
    ("à dạ em cứ tư vấn đi",                                1, "filler_complete"),
    ("ờ ừ tôi hiểu rồi",                                    1, "filler_complete"),
    ("ờ cái đó tôi không cần",                              1, "filler_complete"),
    ("à thôi không sao",                                    1, "filler_complete"),
    ("ừ anh đặt lịch ngày mai nhé",                         1, "filler_complete"),

    # ══════════════════════════════════════════════════════════════
    # 5. CALL CENTER — inbound khiếu nại
    # ══════════════════════════════════════════════════════════════
    ("Mạng nhà tôi bị mất kết nối từ sáng đến giờ.",        1, "callcenter_complaint"),
    ("Tôi đã gọi 3 lần rồi mà vẫn chưa được giải quyết.",  1, "callcenter_complaint"),
    ("Hóa đơn tháng này sai so với thực tế sử dụng.",       1, "callcenter_complaint"),
    ("Nhân viên kỹ thuật hẹn 2 giờ chiều mà chưa thấy đến.", 1, "callcenter_complaint"),
    ("Tôi muốn khiếu nại về thái độ phục vụ.",              1, "callcenter_complaint"),
    ("Tôi bị trừ tiền hai lần cho cùng một giao dịch.",     1, "callcenter_complaint"),
    ("Sản phẩm bị lỗi ngay từ khi mua về.",                 1, "callcenter_complaint"),
    ("Tôi yêu cầu hoàn tiền trong vòng",                    0, "callcenter_complaint"),
    ("Vấn đề của tôi là",                                   0, "callcenter_complaint"),
    ("Tôi gọi để phản ánh về",                              0, "callcenter_complaint"),

    # ══════════════════════════════════════════════════════════════
    # 6. KHẢO SÁT CSAT / NPS
    # ══════════════════════════════════════════════════════════════
    ("Tôi cho 9 điểm.",                                     1, "survey"),
    ("Dịch vụ rất tốt, tôi hài lòng.",                      1, "survey"),
    ("Ổn thôi, không có gì đặc biệt.",                      1, "survey"),
    ("Tôi sẽ giới thiệu cho bạn bè.",                       1, "survey"),
    ("Thái độ nhân viên cần cải thiện hơn.",                 1, "survey"),
    ("8 điểm.",                                             1, "survey"),
    ("Tôi không hài lòng lắm.",                             1, "survey"),
    ("dạ em thấy dịch vụ cũng ổn lắm",                      1, "survey"),
    ("tôi cho 8 điểm",                                      1, "survey"),
    ("ờ anh thấy cũng tạm được",                            1, "survey"),
    ("Điểm của tôi là",                                     0, "survey"),
    ("Tôi đánh giá dịch vụ là",                             0, "survey"),
    ("Về chất lượng thì tôi thấy",                          0, "survey"),
    ("Nhân viên hỗ trợ rất nhiệt tình, tuy nhiên",         0, "survey"),

    # ══════════════════════════════════════════════════════════════
    # 7. TELESALE — từ chối / đồng ý
    # ══════════════════════════════════════════════════════════════
    ("Thôi không cần đâu em ơi.",                           1, "telesale"),
    ("Anh không có nhu cầu.",                               1, "telesale"),
    ("Để anh nghĩ thêm đã.",                                1, "telesale"),
    ("Gửi thông tin qua zalo cho anh nhé.",                 1, "telesale"),
    ("Tôi đang바쁘, gọi lại sau được không?",               1, "telesale"),
    ("Mức phí đó hơi cao so với ngân sách của tôi.",        1, "telesale"),
    ("Ok, anh đăng ký thử xem sao.",                        1, "telesale"),
    ("Không cần đâu em.",                                   1, "telesale"),
    ("Anh đang bận, gọi lại",                               0, "telesale"),
    ("Mức phí đó thì",                                      0, "telesale"),
    ("Anh cần suy nghĩ thêm về",                            0, "telesale"),
    ("Nếu có ưu đãi thêm thì",                              0, "telesale"),

    # ══════════════════════════════════════════════════════════════
    # 8. CÂU NGẮN ĐẶC BIỆT
    # ══════════════════════════════════════════════════════════════
    ("Dạ.",                                                 1, "short"),
    ("Vâng.",                                               1, "short"),
    ("Ừ.",                                                  1, "short"),
    ("Ok.",                                                 1, "short"),
    ("Được.",                                               1, "short"),
    ("Không.",                                              1, "short"),
    ("Thôi.",                                               1, "short"),
    ("Rồi.",                                                1, "short"),
    ("Ừm.",                                                 0, "short"),
    ("À.",                                                  0, "short"),
    ("Ờ.",                                                  0, "short"),

    # ══════════════════════════════════════════════════════════════
    # 9. NGẬP NGỪNG GIỮA CÂU
    # ══════════════════════════════════════════════════════════════
    ("Ờ, tôi muốn, ờ, hỏi về",                             0, "hesitation"),
    ("Thì là, à, tôi cần",                                  0, "hesitation"),
    ("Số điện thoại à, là 0912",                            0, "hesitation"),
    ("Tôi, ừm, muốn đặt lịch cho, à, ngày",                0, "hesitation"),
    ("Ờ thì, anh muốn hỏi, là cái gói đó",                 0, "hesitation"),
    ("Ừ thì tôi cũng không, ờ",                             0, "hesitation"),

    # ══════════════════════════════════════════════════════════════
    # 10. CÂU DÀI PHỨC TẠP
    # ══════════════════════════════════════════════════════════════
    ("Tôi đã sử dụng dịch vụ của công ty được 3 năm và chưa bao giờ gặp vấn đề gì.", 1, "long_complete"),
    ("Nếu giá không giảm thêm thì tôi sẽ cân nhắc chuyển sang nhà cung cấp khác.",   1, "long_complete"),
    ("Mã đơn hàng của tôi là DH2024-00123456, tôi muốn kiểm tra tình trạng giao hàng.", 1, "long_complete"),
    ("Dạ anh ơi, em có thể gặp trực tiếp nhân viên tư vấn tại văn phòng không ạ?",  1, "long_complete"),
    ("Tôi đã thanh toán rồi nhưng hệ thống vẫn báo chưa, tôi cần xử lý ngay hôm nay.", 1, "long_complete"),
    ("Tôi đã gọi điện nhiều lần nhưng mà vấn đề vẫn chưa",                           0, "long_incomplete"),
    ("Nếu như công ty có thể hỗ trợ thêm về",                                        0, "long_incomplete"),
    ("Mã số hợp đồng của tôi là, ờ, để tôi tìm",                                     0, "long_incomplete"),
    ("Tôi muốn hỏi về chính sách hoàn trả, cụ thể là trong trường hợp",              0, "long_incomplete"),

    # ══════════════════════════════════════════════════════════════
    # 11. EDGE CASES — dễ nhầm lẫn
    # ══════════════════════════════════════════════════════════════
    ("tôi thấy cũng được rồi.",                             1, "edge_case"),  # "được rồi" = hoàn chỉnh
    ("à không, ý tôi muốn nói là",                         0, "edge_case"),  # tự sửa lại, chưa xong
    ("Dịch vụ ok, nhưng giá hơi cao.",                      1, "edge_case"),  # nhưng + ý hoàn chỉnh
    ("Thôi thì tùy em.",                                    1, "edge_case"),  # thì + ý hoàn chỉnh
    ("Anh không biết nữa.",                                 1, "edge_case"),  # hoàn chỉnh
    ("Để anh hỏi lại vợ đã rồi tính.",                     1, "edge_case"),  # hoàn chỉnh
    ("Cũng được.",                                          1, "edge_case"),  # hoàn chỉnh — "cũng được" là cụm
    ("Cũng",                                               0, "edge_case"),  # chỉ 1 từ
    ("Ừ cũng được.",                                        1, "edge_case"),  # "cũng được" = cụm hoàn chỉnh
    ("Giá như vậy thì",                                     0, "edge_case"),  # thì = dangling
    ("Thì ra là vậy.",                                      1, "edge_case"),  # "thì ra là vậy" = hoàn chỉnh
    ("Được thì được nhưng",                                 0, "edge_case"),  # nhưng = dangling
    ("Tôi thấy ổn.",                                        1, "edge_case"),
    ("Nếu được thì tốt.",                                   1, "edge_case"),  # nếu đầu câu, ý hoàn chỉnh

    # ══════════════════════════════════════════════════════════════
    # 12. "LÀ" + GIÁ TRỊ ĐẦY ĐỦ — phải là COMPLETE
    # ══════════════════════════════════════════════════════════════
    ("Tên tôi là Trần Thị Mai.",                            1, "la_with_value"),
    ("Địa chỉ của tôi là 45 Nguyễn Huệ, quận 1.",          1, "la_with_value"),
    ("Số CMND của tôi là 079123456789.",                    1, "la_with_value"),
    ("Mã đơn hàng là ORD-20241501.",                        1, "la_with_value"),
    ("Email của tôi là abc@gmail.com.",                     1, "la_with_value"),
    ("Ngày sinh là 20 tháng 5 năm 1988.",                   1, "la_with_value"),
    ("Số tài khoản là 0123456789.",                         1, "la_with_value"),
    ("Biển số xe là 51A-12345.",                            1, "la_with_value"),
    ("Mã khách hàng là KH-00456.",                          1, "la_with_value"),
    ("Họ tên của tôi là Lê Quang Vinh.",                    1, "la_with_value"),
    # Incomplete — chưa có giá trị
    ("Họ tên là",                                           0, "la_with_value"),
    ("Địa chỉ nhà tôi là",                                  0, "la_with_value"),
    ("Số hợp đồng của tôi là",                              0, "la_with_value"),
    ("Email của tôi là",                                    0, "la_with_value"),

    # ══════════════════════════════════════════════════════════════
    # 13. "ĐỂ ANH/EM... ĐÃ" — hoãn lại, COMPLETE
    # ══════════════════════════════════════════════════════════════
    ("Để anh nghĩ thêm đã.",                                1, "de_postpone"),
    ("Để tôi hỏi lại vợ đã rồi tính.",                     1, "de_postpone"),
    ("Để em kiểm tra lại đã nhé.",                          1, "de_postpone"),
    ("Để anh xem lịch đã.",                                 1, "de_postpone"),
    ("Để tôi hỏi sếp đã rồi báo em.",                      1, "de_postpone"),
    ("Cho anh nghĩ thêm chút.",                             1, "de_postpone"),
    ("Thôi để anh suy nghĩ thêm vậy.",                     1, "de_postpone"),
    ("Cho tôi xem lại đã.",                                 1, "de_postpone"),
    ("Để em hỏi lại kỹ thuật đã.",                          1, "de_postpone"),
    # Incomplete
    ("Để anh nghĩ thêm về",                                 0, "de_postpone"),
    ("Cho tôi hỏi thêm về vấn đề",                         0, "de_postpone"),
    ("Để em kiểm tra về",                                   0, "de_postpone"),

    # ══════════════════════════════════════════════════════════════
    # 14. "NHƯNG" GIỮA CÂU — ý hoàn chỉnh sau nhưng → COMPLETE
    # ══════════════════════════════════════════════════════════════
    ("Dịch vụ ok nhưng giá hơi cao.",                       1, "nhung_middle"),
    ("Nhân viên nhiệt tình nhưng chờ lâu quá.",             1, "nhung_middle"),
    ("Tôi hài lòng nhưng muốn hoàn tiền phần chênh lệch.", 1, "nhung_middle"),
    ("Gói này tốt nhưng tôi không có nhu cầu.",             1, "nhung_middle"),
    ("Sản phẩm đẹp nhưng giao hàng chậm hơn dự kiến.",     1, "nhung_middle"),
    ("Mạng ổn nhưng hay bị ngắt vào ban đêm.",              1, "nhung_middle"),
    ("Tôi đồng ý nhưng cần xem lại điều khoản.",           1, "nhung_middle"),
    ("Giá tốt nhưng chất lượng chưa đạt kỳ vọng.",         1, "nhung_middle"),
    # nhưng ở cuối → INCOMPLETE
    ("Tôi muốn mua nhưng",                                  0, "nhung_middle"),
    ("Dịch vụ tốt nhưng",                                   0, "nhung_middle"),
    ("Sản phẩm ok nhưng",                                   0, "nhung_middle"),

    # ══════════════════════════════════════════════════════════════
    # 15. "THÌ RA / À RA LÀ" — câu nhận ra, COMPLETE
    # ══════════════════════════════════════════════════════════════
    ("Thì ra là vậy.",                                      1, "thi_ra"),
    ("À ra là thế.",                                        1, "thi_ra"),
    ("Ồ thì ra vậy à.",                                     1, "thi_ra"),
    ("À ra là anh không nhận được thông báo.",              1, "thi_ra"),
    ("Thì ra hệ thống bị lỗi từ hôm qua.",                  1, "thi_ra"),
    ("Ồ ra là phí đó là phí dịch vụ hàng tháng.",          1, "thi_ra"),
    ("À thì ra em gửi nhầm địa chỉ rồi.",                  1, "thi_ra"),
    ("Ồ tôi hiểu rồi, thì ra là vậy.",                     1, "thi_ra"),
    # Incomplete
    ("Thì ra là",                                           0, "thi_ra"),
    ("À ra thì",                                            0, "thi_ra"),

    # ══════════════════════════════════════════════════════════════
    # 16. "VẬY" KẾT CÂU — tiểu từ kết thúc, COMPLETE
    # ══════════════════════════════════════════════════════════════
    ("Thôi anh gọi lại sau vậy.",                           1, "vay_final"),
    ("Tôi đặt lịch ngày mai vậy nhé.",                      1, "vay_final"),
    ("Thôi để vậy đi.",                                     1, "vay_final"),
    ("Anh cần suy nghĩ thêm vậy.",                          1, "vay_final"),
    ("Ừ thôi vậy nhé.",                                     1, "vay_final"),
    ("Vậy thôi anh không mua nữa.",                         1, "vay_final"),
    ("Thôi vậy em ơi.",                                     1, "vay_final"),
    ("Tôi hiểu rồi, vậy nhé.",                              1, "vay_final"),
    # Incomplete — vậy ở giữa
    ("Nếu vậy thì",                                         0, "vay_final"),
    ("Vậy thì anh cần",                                     0, "vay_final"),
    ("Vậy mà tôi vẫn chưa",                                 0, "vay_final"),

    # ══════════════════════════════════════════════════════════════
    # 17. GIỌNG NAM BỘ — "dạ/ạ/nhen/nghen/hen"
    # ══════════════════════════════════════════════════════════════
    ("Dạ anh ơi, em gọi để hỏi về gói cước.",               1, "southern_dialect"),
    ("Ừ thôi được nhen anh.",                               1, "southern_dialect"),
    ("Vậy nghen, em cảm ơn anh.",                           1, "southern_dialect"),
    ("Thôi được rồi nhen.",                                 1, "southern_dialect"),
    ("Dạ em hiểu rồi ạ.",                                   1, "southern_dialect"),
    ("Không cần đâu anh ơi, thôi.",                         1, "southern_dialect"),
    ("Dạ để em xem lại đã nghen.",                          1, "southern_dialect"),
    ("Ừ thì dạ, anh đồng ý.",                               1, "southern_dialect"),
    # Incomplete
    ("Dạ thì anh muốn",                                     0, "southern_dialect"),
    ("Ừ mà vấn đề là",                                      0, "southern_dialect"),
    ("Dạ thì cái đó",                                       0, "southern_dialect"),

    # ══════════════════════════════════════════════════════════════
    # 18. CUNG CẤP THÔNG TIN SỐ — số điện thoại, ngày, tiền
    # ══════════════════════════════════════════════════════════════
    ("Số điện thoại của tôi là 0901 234 567.",              1, "provide_info"),
    ("Tôi sinh ngày 15 tháng 8 năm 1990.",                  1, "provide_info"),
    ("Số tiền tôi cần hoàn là 500 nghìn đồng.",             1, "provide_info"),
    ("Mã OTP của tôi là 123456.",                            1, "provide_info"),
    ("Tôi muốn chuyển khoản 2 triệu.",                      1, "provide_info"),
    ("Hợp đồng số BH-2024-0045678.",                        1, "provide_info"),
    ("Tôi cần thanh toán trước ngày 30 tháng này.",         1, "provide_info"),
    # Incomplete — đang đọc số
    ("Số điện thoại là 0912",                               0, "provide_info"),
    ("Mã giao dịch của tôi là",                             0, "provide_info"),
    ("Tôi muốn chuyển khoản số tiền",                       0, "provide_info"),

    # ══════════════════════════════════════════════════════════════
    # 19. TỰ SỬA / NÓI LẠI — restart giữa câu
    # ══════════════════════════════════════════════════════════════
    ("Ý tôi là, à không, tôi muốn nói là",                 0, "self_correction"),
    ("Không không, ý tôi muốn hỏi về",                     0, "self_correction"),
    ("À ý tôi là vấn đề với hóa đơn, à không phải",        0, "self_correction"),
    ("Dạ ý em là, à thôi để em nói lại,",                  0, "self_correction"),
    ("Tôi muốn, à không, tôi cần hủy",                     0, "self_correction"),
    # Tự sửa nhưng kết thúc xong
    ("Không, ý tôi là tôi muốn đổi sản phẩm.",             1, "self_correction"),
    ("À không, anh muốn hủy đơn hàng đó.",                 1, "self_correction"),
    ("Ý tôi là gói cước hàng tháng, không phải năm.",       1, "self_correction"),

    # ══════════════════════════════════════════════════════════════
    # 20. YÊU CẦU HỖ TRỢ / HỎI THÔNG TIN
    # ══════════════════════════════════════════════════════════════
    ("Tôi cần hỗ trợ khẩn cấp.",                           1, "support_request"),
    ("Cho tôi gặp người có thẩm quyền giải quyết.",         1, "support_request"),
    ("Tôi muốn biết thời gian xử lý khiếu nại.",            1, "support_request"),
    ("Bao giờ thì kỹ thuật đến nhà tôi?",                   1, "support_request"),
    ("Làm sao để tôi theo dõi trạng thái đơn hàng?",        1, "support_request"),
    ("Phí hủy hợp đồng là bao nhiêu?",                      1, "support_request"),
    ("Tôi muốn nâng cấp gói dịch vụ lên gói cao hơn.",     1, "support_request"),
    # Incomplete
    ("Tôi cần hỗ trợ về vấn đề",                           0, "support_request"),
    ("Cho tôi hỏi thêm về",                                 0, "support_request"),
    ("Bao giờ thì",                                         0, "support_request"),
    ("Làm sao để",                                          0, "support_request"),
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests(model_dir: str, threshold: float, show_errors: bool = False):
    import torch
    import torch.nn.functional as F
    import re
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _DANGLING_RE = re.compile(
        r"""(?ix)\b(cũng|thì|là|và|hoặc|nhưng|mà|để|vì|bởi|rằng|hay|nếu|khi|sẽ|bị|nên|vẫn|chỉ|tuy\s+nhiên|trong\s+vòng|bởi\s+vì|chính\s+là)\s*[.!?,;]?\s*$""",
        re.UNICODE,
    )
    _ACK_RE = re.compile(
        r"""(?ix)^\s*(dạ|vâng|ừ|ok|okay|rồi|thôi|xong|được|cũng\s+được|thôi\s+được|ừ\s+được|ừ\s+cũng\s+được|dạ\s+rồi|vâng\s+rồi|ok\s+rồi|thôi\s+thì\s+thôi|không\s+cần|không\s+sao|không\s+cần\s+đâu)\s*[.!,]?\s*$""",
        re.UNICODE,
    )

    print(f"Loading model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    print(f"Threshold: {threshold}\n")

    results_by_cat = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    all_correct = 0

    with torch.no_grad():
        for raw_text, expected, category in TEST_CASES:
            text = clean_asr(raw_text)   # simulate ASR output — no punctuation
            enc = tokenizer(text, max_length=128, padding="max_length",
                            truncation=True, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"]).logits
            probs = F.softmax(logits, dim=-1).squeeze().tolist()
            p_complete = probs[1]

            overridden = None
            if _ACK_RE.match(text):
                p_complete = max(probs[1], threshold + 0.01)
                overridden = "ack_word"
            elif p_complete >= threshold and _DANGLING_RE.search(text.strip()):
                p_complete = min(probs[0], threshold - 0.01)
                overridden = "dangling_word"

            predicted = 1 if p_complete >= threshold else 0
            correct = predicted == expected

            results_by_cat[category]["total"] += 1
            if correct:
                results_by_cat[category]["correct"] += 1
                all_correct += 1
            else:
                results_by_cat[category]["errors"].append({
                    "text": text,
                    "raw": raw_text if raw_text != text else None,
                    "expected": "COMPLETE" if expected == 1 else "INCOMPLETE",
                    "predicted": "COMPLETE" if predicted == 1 else "INCOMPLETE",
                    "prob_complete": round(p_complete, 4),
                    "overridden": overridden or False,
                })

    total = len(TEST_CASES)
    print("=" * 70)
    print(f" KẾT QUẢ TỔNG QUAN: {all_correct}/{total} đúng  "
          f"({100*all_correct/total:.1f}%)")
    print("=" * 70)

    # Per-category report
    cat_order = [
        "basic_complete", "basic_incomplete", "dangling_word",
        "filler_complete", "callcenter_complaint", "survey",
        "telesale", "short", "hesitation", "long_complete",
        "long_incomplete", "edge_case",
        "la_with_value", "de_postpone", "nhung_middle",
        "thi_ra", "vay_final", "southern_dialect",
        "provide_info", "self_correction", "support_request",
    ]
    cat_labels = {
        "basic_complete":       "Câu hoàn chỉnh — cơ bản",
        "basic_incomplete":     "Câu chưa xong  — cơ bản",
        "dangling_word":        "Dangling word (cũng/thì/là...)",
        "filler_complete":      "Ờ/ừ/à + ý hoàn chỉnh",
        "callcenter_complaint": "Call center — khiếu nại",
        "survey":               "Khảo sát CSAT/NPS",
        "telesale":             "Telesale",
        "short":                "Câu cực ngắn",
        "hesitation":           "Ngập ngừng giữa câu",
        "long_complete":        "Câu dài — hoàn chỉnh",
        "long_incomplete":      "Câu dài — chưa xong",
        "edge_case":            "Edge cases",
        "la_with_value":        "Là + giá trị đầy đủ",
        "de_postpone":          "Để anh/em... đã (hoãn lại)",
        "nhung_middle":         "Nhưng ở giữa câu",
        "thi_ra":               "Thì ra / À ra là",
        "vay_final":            "Vậy kết câu",
        "southern_dialect":     "Giọng Nam (nhen/nghen/dạ)",
        "provide_info":         "Cung cấp số/thông tin",
        "self_correction":      "Tự sửa giữa câu",
        "support_request":      "Yêu cầu hỗ trợ",
    }

    print()
    print(f"{'Nhóm':<38} {'Đúng':>6} {'Tổng':>6} {'Tỉ lệ':>8}")
    print("-" * 62)
    for cat in cat_order:
        if cat not in results_by_cat:
            continue
        r = results_by_cat[cat]
        pct = 100 * r["correct"] / r["total"]
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        label = cat_labels.get(cat, cat)
        ok = "✓" if pct == 100 else ("~" if pct >= 80 else "✗")
        print(f"{ok} {label:<36} {r['correct']:>6}/{r['total']:<6} {pct:>6.1f}%  {bar}")

    # Error details
    all_errors = [
        (cat, e)
        for cat in cat_order
        if cat in results_by_cat
        for e in results_by_cat[cat]["errors"]
    ]

    if all_errors:
        print(f"\n{'─'*70}")
        print(f" CÁC CASE SAI ({len(all_errors)} lỗi):")
        print(f"{'─'*70}")
        for cat, err in all_errors:
            override_note = f" [{err['overridden']}]" if err["overridden"] else ""
            raw_note = f"  (raw: \"{err['raw']}\")" if err.get("raw") else ""
            print(f"\n  [{cat_labels.get(cat, cat)}]")
            print(f"  ASR text : \"{err['text']}\"{raw_note}")
            print(f"  Expected : {err['expected']}")
            print(f"  Predicted: {err['predicted']}{override_note}  (P(complete)={err['prob_complete']})")
    else:
        print(f"\n{'─'*70}")
        print(" Không có case nào sai! 🎯")

    print(f"\n{'='*70}")
    print(f" TỔNG: {all_correct}/{total} đúng  |  Sai: {total-all_correct}  |  Accuracy: {100*all_correct/total:.1f}%")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",  default="runs/sbd_model/final")
    parser.add_argument("--threshold",  type=float, default=0.65)
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    run_tests(args.model_dir, args.threshold, args.show_errors)


if __name__ == "__main__":
    main()
