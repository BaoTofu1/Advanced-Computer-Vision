# Model B1/B2 của Trí

Lượt train song song do Trí chạy trên GPU server (Linux), cùng cấu hình với lượt của Bảo: fine-tune toàn bộ InceptionResnetV1 (VGGFace2), 1.600 người DigiFace, 5 epoch, batch 8, AdamW (encoder 1e-5, head 1e-3, weight decay 1e-4), AMP, seed 20261028. Bảng chia dữ liệu giống hệt của nhóm (SHA khác chỉ vì kiểu xuống dòng Linux).

| Thư mục | Nội dung |
|---|---|
| `training/` | `train.py` (script train + đánh giá validation), `README_TRAINING.md`, lệnh đã chạy (`commands.sh`), môi trường conda |
| `logs/` | Log B0, B1, B2 trên server |
| `experiments/digiface_validation/` | Cấu hình, lịch sử từng epoch, kết quả validation tốt nhất |
| `checkpoints/` | `B1_Tri_encoder.pt`, `B2_Tri_encoder.pt`: chỉ encoder, tách từ `best.pt` 292 MB (bỏ head, optimizer, RNG). Nạp bằng `tai_model.load_encoder(checkpoint=...)` |
| `So_sanh_model_Tri_va_Bao.ipynb` | Chấm cả 5 model bằng cùng evaluator trên DigiFace validation/test, ROF, MFR2 |

Khác biệt với lượt của Bảo: che của B2 là hình chữ nhật đen với tọa độ riêng (Bảo dùng thuật toán augmentation của nhóm: màu xám ngẫu nhiên); code và máy khác; epoch được chọn B1 = 4, B2 = 5.

## Kết quả (cùng evaluator)

| | B0 | B1 Bảo | B1 Trí | B2 Bảo | B2 Trí |
|---|---:|---:|---:|---:|---:|
| DigiFace val, DIR nền | 57,1% | 93,5% | 94,8% | 93,5% | 94,6% |
| DigiFace test, DIR nền | 62,9% | 95,4% | 96,9% | 95,2% | 95,8% |
| DigiFace test, DIR query che | 25,5% | 59,0% | 63,8% | 81,1% | 75,0% |
| ROF (ảnh thật), Rank-1 | **92,8%** | 72,2% | 68,9% | 67,2% | 60,3% |
| ROF khẩu trang, Rank-1 | **80,3%** | 34,8% | 30,9% | 36,5% | 30,3% |
| MFR2, AUC | **0,991** | 0,935 | 0,915 | 0,944 | 0,942 |

- Evaluator của nhóm tái lập được số Trí báo trên validation (B1: 3.506 so với 3.507 query đúng; B2: 3.499 so với 3.498; lệch 1 query do làm tròn GPU khác máy).
- Hai lượt train cho cùng bức tranh: tốt trên DigiFace, kém B0 rõ rệt trên ảnh thật. Model của Trí còn kém B0 nhiều hơn trên ROF (B2 Trí − B0 = −32,5 điểm Rank-1, CI 95% [−36,0; −29,2]).
- B2 Bảo hơn B2 Trí trên query che tổng hợp (−6,1 điểm, CI [−7,9; −4,2]), có thể do che xám ngẫu nhiên đa dạng hơn che đen; cả hai B2 đều không giúp trên khẩu trang thật.

Ngưỡng hai model Trí chọn trên DigiFace validation theo quy tắc chung và khóa trong `khoa_nguong_model_Tri.json` trước khi chấm test.
