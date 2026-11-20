# Bước 6C — Train full B1/B2 và model đã lưu

Cập nhật 06/11/2026. Notebook [Train_full_B1_B2.ipynb](Train_full_B1_B2.ipynb) train trên toàn bộ 1.600 người DigiFace train (57.600 ảnh/epoch), theo cấu hình của lượt train song song trong nhóm: fine-tune toàn bộ InceptionResnetV1, 5 epoch, batch 8, AdamW (encoder 1e-5, classifier 1e-3, weight decay 1e-4), FP16, seed 20261028. B2 che 50% lượt ảnh bằng đúng thuật toán của [notebook augmentation](../augmentation/Augmentation_che_co_kiem_soat.ipynb). Mỗi nhánh 36.000 bước optimizer, khoảng 10 phút/epoch trên RTX 4060 Laptop.

## Kết quả trên DigiFace validation

100 người quen + 100 người lạ chưa xuất hiện lúc train; 3.700 query người quen, 4.000 query người lạ. Ngưỡng của mỗi model chọn trên ảnh nền ở FPIR ≤ 5%; query che dùng nguyên ngưỡng đó.

| Model | Epoch chọn | DIR ảnh nền | Rank-1 | DIR query che | FPIR query che |
|---|---:|---:|---:|---:|---:|
| B0 (pretrained) | – | 57,16% | 82,86% | 22,73% | 2,75% |
| B1 (ảnh nền) | 3 | 93,49% | 97,00% | 56,51% | 4,33% |
| B2 (ảnh nền + che) | 4 | 93,49% | 97,08% | **79,62%** | 4,70% |

- Ảnh nền: B1 và B2 bằng nhau (cùng 3.459/3.700 query đúng). B2 − B1 = 0,00 điểm, CI 95% [−1,00; +1,00].
- Query che: B2 − B1 = **+23,11 điểm**, CI 95% [+20,76; +25,68]. Thêm che lúc train không làm giảm kết quả trên ảnh nền.
- Theo vùng che (DIR B0 / B1 / B2): nửa dưới 8,2 / 20,1 / 57,6%; trái 40,9 / 83,8 / 91,3%; phải 41,6 / 84,3 / 91,5%; mắt 1,3 / 39,4 / 78,7%.

CI là bootstrap theo người, 2.000 lần, tại checkpoint và ngưỡng đã chọn; mới có một seed. Che ở đây là hình chữ nhật tổng hợp trên ảnh tổng hợp, nên chưa nói được gì về khẩu trang thật hay webcam. Đó là việc của bước 7 (ROF/MFR2) và bước 8.

## Đối chiếu với lượt train song song

| Model | DIR lượt song song | DIR notebook này | Chênh |
|---|---:|---:|---:|
| B0 | 57,16% | 57,16% | 0,00 |
| B1 | 94,78% | 93,49% | −1,29 |
| B2 | 94,54% | 93,49% | −1,05 |

B0 trùng khớp, nên hai bên chấm cùng một bộ query. B1/B2 lệch khoảng 1 điểm, nằm trong CI của từng model (B1: 91,6–95,3%; B2: 91,6–95,2%). Độ lệch này hợp lý với thứ tự ngẫu nhiên khác nhau giữa hai lượt và cách chọn epoch chưa rõ của lượt kia. Lượt song song chưa có kết quả trên query che; đó là phần cần hỏi thêm.

## Model đã lưu và cách dùng lại

| File | Dùng cho |
|---|---|
| `checkpoints_full/B1_best_encoder.pt`, `B2_best_encoder.pt` | Encoder của epoch được chọn (~90 MB/file). Dùng cho bước 7, demo webcam |
| `checkpoints_full/B1_last_resume.pt`, `B2_last_resume.pt` | Trạng thái đầy đủ để train tiếp; chỉ có trên máy đã train, không có trong ZIP |
| `tai_model.py` | `load_encoder('B2')` và `embed(model, anh_mat)`, không cần train lại |

```python
import sys; sys.path.insert(0, 'ban_giao_nhom/huan_luyen')
from tai_model import load_encoder, embed
model = load_encoder('B2')           # 'B0', 'B1' hoặc 'B2'
vectors = embed(model, anh_mat)      # (N, 512), đã chuẩn hóa L2
```

Mục 6 của notebook nạp lại hai file encoder bằng chính `tai_model.py`, chấm lại từ đầu và ra đúng từng số đếm lúc train.

**Chạy lại notebook không train lại.** Mặc định (`CV_TRAIN_MODE=auto`) notebook kiểm dấu vân tay (cấu hình, danh sách ảnh, thuật toán che, khởi tạo, code train): khớp thì nạp model đã lưu và chỉ chấm lại (vài phút); train dở thì train tiếp từ epoch cuối đã lưu. Muốn train lại từ đầu thì đặt biến môi trường `CV_TRAIN_MODE=retrain` trước khi mở kernel. Lượt chạy lưu trong notebook này đã thật sự dùng tính năng train tiếp: máy bị tắt sau epoch 2 của B1, lần chạy sau ghi `resumed_after_epoch_2`.

## Chạy lại

1. Cần GPU NVIDIA, môi trường theo [hướng dẫn cài đặt](../README_GOI_BAN_GIAO.md) và ảnh DigiFace ở `data/raw/digiface/images_eda/` (80.000 ảnh). Notebook băm SHA từng ảnh và dừng nếu khác bảng protocol.
2. Mở notebook, Restart Kernel → Run All. Lần đầu tạo cache ảnh ~5,5 GB trong `full_outputs/cache/` (khoảng 2–3 phút) và nạp ~5,5 GB vào RAM; máy cần ít nhất 16 GB RAM.
3. Có sẵn file `*_best_encoder.pt` nhưng không có `*_last_resume.pt` (trường hợp giải nén từ ZIP) thì notebook sẽ train lại. Muốn chỉ dùng model, dùng `tai_model.py`.

Bảng đầu ra ở `full_outputs/tables/`, hình cho slide ở `full_outputs/figures/`, tóm tắt lượt chạy ở `full_outputs/execution_summary.json`.
