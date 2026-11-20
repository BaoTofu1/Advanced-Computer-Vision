# Bước 7 — Đánh giá độc lập B0/B1/B2

Notebook [Danh_gia_doc_lap_B0_B1_B2.ipynb](Danh_gia_doc_lap_B0_B1_B2.ipynb) chấm ba model đã chốt ở bước 6 trên ba bộ chưa dùng để train hay chọn tham số. Model, ngưỡng (từ DigiFace validation), cách tiền xử lý, cách sinh che và cách tính chỉ số được ghi vào [`khoa_cau_hinh_buoc7.json`](khoa_cau_hinh_buoc7.json) trước khi mở ảnh test (11/11/2026 13:56). Chạy lại thì notebook chỉ kiểm file khóa, không ghi đè.

## Kết quả chính

| | B0 | B1 | B2 |
|---|---:|---:|---:|
| DigiFace test, ảnh nền — DIR tại ngưỡng validation | 62,92% | 95,38% | 95,24% |
| DigiFace test, query che nhân tạo — DIR | 25,46% | 58,95% | **81,05%** |
| DigiFace test — FPIR thực tế tại ngưỡng validation | 4,90% | 6,10% | 8,72% |
| ROF (ảnh thật), mọi query — Rank-1 | **92,80%** | 72,23% | 67,16% |
| ROF, rõ mặt — Rank-1 | **97,77%** | 86,85% | 82,03% |
| ROF, khẩu trang — Rank-1 | **80,34%** | 34,83% | 36,52% |
| ROF, kính râm — Rank-1 | **84,99%** | 49,37% | 43,02% |
| ROF, mọi query — FPIR tại ngưỡng validation | 0,09% | 89,64% | 93,46% |
| MFR2 (848 cặp) — AUC | **0,9908** | 0,9349 | 0,9441 |
| MFR2 — EER | **3,30%** | 14,86% | 13,44% |

ROF dùng cách A (resize crop có sẵn). CI 95% bootstrap theo người, ghép cặp giữa model:

- DigiFace test, query che: B2 − B1 = +22,11 điểm [+19,86; +24,51].
- ROF Rank-1, mọi query: B1 − B0 = −20,57 điểm [−23,40; −17,78]; B2 − B0 = −25,63 [−29,14; −22,25]; B2 − B1 = −5,07 [−6,94; −3,23].
- ROF Rank-1, khẩu trang (178 query, 47 người): B2 − B1 = +1,69 [−10,29; +13,97], không phân biệt được.
- MFR2 AUC: B1 − B0 = −0,056 [−0,070; −0,042]; B2 − B0 = −0,047 [−0,061; −0,033].

Cách B (chạy lại MTCNN như ứng dụng webcam) cho cùng kết luận. MTCNN thấy đúng một mặt ở 96,1% ảnh ROF rõ mặt, 93,3% kính râm nhưng chỉ 72,7% ảnh đeo khẩu trang.

## Đọc kết quả

1. **Trên ảnh tổng hợp, kết quả bước 6 giữ được trên 200 người mới.** B2 vẫn hơn B1 khoảng 22 điểm trên ảnh che nhân tạo và ngang B1 trên ảnh nền.
2. **Trên ảnh người thật, fine-tune làm model kém đi.** B1/B2 thua B0 ở mọi điều kiện của ROF và trên MFR2. Fine-tune toàn bộ mạng trên khuôn mặt render (DigiFace) làm model học đặc trưng của ảnh tổng hợp và mất một phần khả năng phân biệt mặt thật học từ VGGFace2 (lệch miền + quên kiến thức cũ).
3. **Lợi ích của che nhân tạo không chuyển sang khẩu trang thật.** Trên ROF khẩu trang, B2 không hơn B1 rõ ràng và cả hai kém B0 khoảng 45 điểm Rank-1.
4. **Ngưỡng không chuyển giữa các miền ảnh.** Ngưỡng chọn trên DigiFace làm B1/B2 chấp nhận gần hết người lạ trên ảnh thật; ngay cả với B0, ngưỡng DigiFace quá chặt cho ảnh thật (FPIR 0,09%, bỏ sót nhiều người quen). Webcam phải có ngưỡng riêng chọn từ ảnh webcam phiên 2.

## Hệ quả cho bước 8 và báo cáo

- **Demo webcam nên dùng B0** (`load_encoder('B0')` trong `huan_luyen/tai_model.py`), ngưỡng chọn bằng ảnh webcam phiên 2.
- Báo cáo nên trình bày đủ hai mặt: B2 thành công trên ảnh che tổng hợp, nhưng không tổng quát sang ảnh thật. Đây là kết quả âm có giá trị, không phải lỗi thí nghiệm. Các kiểm tra trong notebook (SHA ảnh, hàm chấm ra lại đúng B0 bước 5, file khóa) cho thấy số liệu đúng với cách đo đã định.
- Hướng cải thiện (nếu còn thời gian, ghi vào phần hạn chế/hướng phát triển): train trên ảnh mặt thật (CASIA-WebFace, VGGFace2…), hoặc fine-tune nhẹ (chỉ mở vài lớp cuối, LR nhỏ, ít bước) và chọn checkpoint bằng một tập ảnh thật riêng, không dùng ROF/MFR2.

## Lưu ý về dữ liệu

- ROF/MFR2 là ảnh người nổi tiếng lấy từ web; nhãn chưa được nhóm xác minh độc lập (xem QA). Người lạ đeo khẩu trang trong ROF chỉ có 20 ảnh, nên FPIR riêng nhóm này không đáng tin; bảng có FPIR gộp.
- MFR2 không có fold chính thức nên không báo accuracy 10-fold; AUC/EER không cần ngưỡng. Cặp "rõ–rõ" chỉ có 69 cặp (8 cùng người).
- "DIR tại FPIR 5% trên ROF" dùng ngưỡng chọn trên chính ROF, chỉ để so model, không phải ngưỡng dùng thật.

## Chạy lại

Cần ảnh gốc `data/raw/digiface/images_eda/`, `data/benchmark_views/rof/`, `data/raw/mfr2/`, hai file `huan_luyen/checkpoints_full/*_best_encoder.pt`, và cache embedding B0 bước 5 (`danh_gia_b0/notebook_outputs/cache/embeddings.npz`, tạo bằng notebook B0) để kiểm hàm chấm. Restart Kernel → Run All. Lần đầu khoảng 10 phút trên RTX 4060 (phần lớn là MTCNN trên ROF); các lần sau dùng cache, dưới 1 phút. Bảng ở `notebook_outputs/tables/`, hình ở `notebook_outputs/figures/`.
