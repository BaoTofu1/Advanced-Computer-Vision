# QA nhãn / ảnh trùng / bounding box

Mở `QA_nhan_trung_bounding_box.ipynb` để xem đầy đủ code, hình minh họa và diễn giải. Notebook tiếp nối notebook EDA; đã chạy từ kernel mới, 26 cell code, không có lỗi.

## Kết quả chính

- 86.980/86.980 ảnh khớp bytes với lúc làm EDA; đọc lại đủ 10.137 box từ annotation ROF.
- 6.711 crop ROF khớp pixel với annotation (6 box nằm ngoài ảnh đã bị loại từ lúc tải). Khớp pixel chỉ chứng minh crop cắt đúng box, không chứng minh đúng người.
- 5 nhóm ảnh trùng pixel: 1 nhóm khác nhãn người (`kylian_mbappe` / `neymar`), 3 nhóm cùng người nhưng khác điều kiện (clean / sunglasses).
- 488 cặp ảnh gần trùng (pHash) đã được chấm: 96 cặp là cùng một ảnh, 15 cặp chưa chắc, 252 cặp không phải trùng, 120 cặp chỉ là ảnh nền phẳng, 5 cặp trùng pixel.
- 3 file nguồn là logo, không có mặt người; 34 crop sinh từ chúng.
- 723 crop thuộc ảnh có nhiều khuôn mặt nhưng chỉ gắn một tên; chưa biết box nào đúng người.
- 3 ứng viên trùng tên giữa ROF và MFR2: Joe Biden, Justin Trudeau, Emmanuel/Immanuel Macron.

## Đề xuất (chưa áp dụng)

| Trạng thái | Số ảnh |
|---|---:|
| Tạm giữ (hold) | 804 |
| Bản trùng phụ | 82 |
| Cần xem thêm | 2.946 |
| Chưa phát hiện lỗi | 83.148 |

"Chưa phát hiện lỗi" không có nghĩa nhãn đã được xác minh. Nếu áp dụng mức chặt nhất (kịch bản B), ROF khẩu trang còn 198/264 crop và 55 người có cả ảnh rõ lẫn ảnh khẩu trang.

**Lưu ý quan trọng:** phần xem ảnh (62 trang contact sheet, các log review) do trợ lý AI (Codex) thực hiện, chưa có thành viên nào xác nhận. Nhóm cần tự kiểm lại trước khi coi là đã duyệt.

## Các tệp chính

- `notebook_outputs/tables/16_QA_manifest_proposed.csv`: mỗi ảnh một dòng, có lý do, hành động đề xuất và quyết định được đọc từ input riêng; đây là output tự sinh, không sửa tay.
- `notebook_outputs/tables/18_group_decision_board.csv`: 9 quyết định nhóm cần chốt, có cột người phụ trách A/B/C.
- `notebook_outputs/tables/17_scenario_impact.csv`, `17_gallery_feasibility.csv`: mô phỏng hai mức áp dụng.
- `notebook_outputs/tables/15_photo_condition_conflicts.csv`, `12_alias_mapping_candidates.csv`: các xung đột điều kiện và trùng tên cần chốt.
- `notebook_outputs/contact_sheets/`: trang ảnh để duyệt; `14_all_review_page_map.csv` nối từng ảnh/cặp với trang.

## Cập nhật khi nhóm duyệt

Quyết định theo ảnh được ghi vào **`group_decisions.csv`**, không ghi vào manifest trong `notebook_outputs/`. Notebook chỉ đọc input này và giữ nguyên khi Run All; phần header-only hiện tại nghĩa là **chưa có quyết định nhóm nào**.

| Cột | Nội dung bắt buộc |
|---|---|
| `image_id` | ID có trong manifest, duy nhất một dòng cho mỗi ảnh |
| `reviewer_ABC` | Người duyệt thực tế của nhóm |
| `group_decision` | `approve_proposal`, `keep`, `hold` hoặc `exclude` |
| `decision_reason` | Lý do quyết định |
| `reviewed_at` | Ngày ISO 8601, dạng `YYYY-MM-DD` |
| `evidence_reference` | Tên trang/case/tệp hoặc nguồn bằng chứng đã xem |

`approve_proposal` là đồng ý hành động đề xuất cho ảnh đó. Ô rỗng không phải xác nhận; ID lạ, ID lặp, thiếu trường hoặc quyết định sai làm notebook dừng để sửa input. Khi đổi ý, cập nhật chính dòng của ảnh trong input rồi Run All lại. Output giữ riêng `qa_action_proposed`, `group_decision` và `decision_status`; đọc quyết định **chưa tự áp dụng** hold/exclude, chưa sửa nhãn hay xác minh danh tính.

Log bằng chứng chi tiết vẫn nằm trong `page_review_log.csv`, `pair_review_decisions.csv`, `case_review_decisions.csv` (người duyệt, ngày, bằng chứng). Không sửa tay bảng trong `notebook_outputs/` vì chúng được dựng lại mỗi lần chạy. Raw, nhãn và split vẫn giữ nguyên.

Restart Kernel → Run All chạy lại logic từ kernel mới. `notebook_outputs/tables/19_scan_provenance.csv` phân biệt `matching_cache` và `computed_now`; kernel mới không đồng nghĩa quét mới toàn bộ ảnh. Đặt `REUSE_SCAN_CACHE=False` nếu cần băm và kiểm crop lại. Summary notebook tự ghi là `notebook_outputs/qa_execution_summary.json`; tệp `qa_notebook_execution_validation.json` là bằng chứng từ runner của một lượt chạy cụ thể, không tự cập nhật khi chạy bằng giao diện Jupyter/VSCode.
