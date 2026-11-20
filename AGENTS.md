# Bản tiếp nối bước 6 — Codex, 05/11/2026

- Đây là bản sao có SHA-256 từ `Đồ Án/Claude Works/TiepNoi_2026-10-28/DoAn_CV_NangCao`, ghi nguồn trong `../CLAUDE_BASELINE_SHA256.json`. Không sửa cây Claude, cây Codex cũ hoặc dự án gốc.
- Quy ước gốc ở `../../../AGENTS.md`; yêu cầu hiện tại của người dùng cho phép sửa khả năng chạy lại, augmentation và pilot train B1/B2. Các giới hạn lịch sử “chỉ QA/chưa preprocessing” đã được yêu cầu mới thay thế.
- Code phân tích/augmentation/training/evaluation chính nằm trong Notebook với Markdown tiếng Việt, kết quả thật, hình và giải thích. Chạy kernel mới tuần tự; không giả execution count/output.
- `data/raw` và `data/benchmark_views` là junction đọc dữ liệu gốc. Không ghi qua junction, không sửa ảnh/nhãn gốc. Output/cache/checkpoint chỉ nằm trong bản tiếp nối này.
- P1–P5 đã được người dùng duyệt ngày 28/10, không hỏi lại. Không coi phê duyệt chính sách là xác minh độc lập mọi nhãn hoặc gallery clean.
- Pilot dùng subset danh tính train và monitor; validation chỉ dùng đánh giá/chọn checkpoint. Không đưa test hoặc ROF/MFR2 vào tuning, không mở ảnh test ở bước này.
- B1/B2 cùng initialization, label mapping, batch/order/optimizer steps và lịch validation. Giữ B1 tên “dữ liệu nền”, không gọi mọi ảnh DigiFace là clean.
- Frozen BatchNorm phải giữ running stats; khi fine-tune kiểm encoder có gradient và checkpoint thực sự đổi, không chỉ classifier. Giữ checkpoint encoder để evaluator người mới không cần classifier.
- Cache cần hash bytes nguồn, cấu hình, code, model và phiên bản thư viện. Phân biệt compute mới/cache. Exception phần mềm phải làm thất bại notebook, không biến thành detector failure.
- CSV score phải round-trip đúng quyết định tại threshold; giữ thống nhất float32 comparison và export đủ chính xác.
