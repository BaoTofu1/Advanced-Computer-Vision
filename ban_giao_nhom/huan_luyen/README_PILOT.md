# Bước 6 — Augmentation và pilot B1/B2

Cập nhật ngày 06/11/2026. Pilot dùng 128 danh tính DigiFace với 4.608 ảnh train và 512 ảnh monitor. B1 dùng ảnh nền. B2 dùng cùng danh sách và thứ tự ảnh, với xác suất thêm che 50% mỗi lượt. Hai nhánh cùng khởi tạo, batch 32 và 432 bước optimizer qua 3 epoch. Epoch 0 warm-up classifier. Epoch 1–2 fine-tune block cuối và projection.

| Notebook đã chạy | Output chính |
|---|---|
| [Augmentation](../augmentation/Augmentation_che_co_kiem_soat.ipynb) | 13.824 lượt ảnh train có đặc tả B2, 7.700 biến thể query cố định, mask minh họa và kiểm tra split |
| [Pilot train B1/B2](Pilot_train_B1_B2.ipynb) | Loss theo epoch, kiểm gradient và BatchNorm, bảng DIR/FPIR, checkpoint encoder và resume |
| [Phân tích và kiểm chứng](Phan_tich_va_kiem_chung_pilot.ipynb) | Kết quả theo bốn vùng che, ví dụ ảnh và phép thử bước tiếp theo sau khi nạp optimizer/RNG |

## Kết quả trên DigiFace validation

Tập validation gồm 300 ảnh gallery nền, 3.700 known query và 4.000 unknown query. Mỗi model chọn ngưỡng trên query nền ở FPIR 5%. Các query che dùng đúng ngưỡng nền của model tương ứng. DIR là tỷ lệ known query vừa được nhận, vừa khớp đúng danh tính. FPIR là tỷ lệ unknown query bị nhận thành người trong gallery.

| Model | Ngưỡng | DIR nền | DIR che | FPIR che |
|---|---:|---:|---:|---:|
| B0 | 0,693074 | 57,16% | 22,73% | 2,75% |
| B1 | 0,631695 | 66,49% | 30,30% | 3,08% |
| B2 | 0,655591 | 66,35% | 31,27% | 3,00% |

B2 so với B1 tăng 0,97 điểm phần trăm DIR trên query che, CI 95% paired bootstrap theo danh tính từ +0,05 đến +1,97 điểm. Trên query nền, chênh lệch là −0,14 điểm, CI từ −1,22 đến +0,95 điểm. Các khoảng này có điều kiện theo checkpoint và ngưỡng đã chọn trên validation. Thử nghiệm dùng một seed và che hình chữ nhật tổng hợp. ROF/MFR2 cùng split DigiFace test được dành cho bước 7.

| Tài nguyên trên RTX 4060 Laptop | Giá trị |
|---|---:|
| Train và validation hai nhánh | 88,1 giây |
| Phần tính toán toàn notebook pilot | 119,1 giây |
| Đỉnh tensor GPU do PyTorch cấp phát trong các epoch | 361,7 MiB |
| Ước lượng phần train của một epoch full warm-up | 0,83 phút |
| Ước lượng phần train của một epoch full fine-tune | 0,94 phút |

Hai ước lượng dùng 57.600 ảnh, batch 32, cùng phần encoder được mở và cache uint8 đã có. Chi phí chuẩn bị cache, validation và ghi checkpoint được tính riêng. Chọn ngân sách full train và số seed trước khi triển khai trên 1.600 danh tính. Sau đó khóa checkpoint, cấu hình và ngưỡng để chạy bước 7.

## Chạy lại bằng VS Code hoặc Jupyter

1. Mở thư mục `DoAn_CV_NangCao` làm thư mục dự án. Chọn kernel Python của môi trường dự án trong **Select Kernel**. Trên máy này, môi trường đã dùng nằm ở `Đồ Án/.venv`. Trên máy khác, tạo môi trường theo [hướng dẫn cài đặt](../README_GOI_BAN_GIAO.md). Lượt pilot đã chạy với Python 3.11.9, PyTorch 2.2.2+cu121 và facenet-pytorch 2.6.0 trên GPU NVIDIA.
2. Kiểm tra `data/raw` và `data/benchmark_views`. Bản làm việc trên máy này dùng junction đến dữ liệu gốc. Sau khi giải nén trên máy khác, đặt dữ liệu đúng cấu trúc manifest. Với các bảng bước 1–5 trong gói, bước 6 đọc ảnh DigiFace ở `data/raw/digiface/images_eda/<identity>/<file>.png`. Chạy lại toàn bộ EDA/QA từ đầu cần thêm dữ liệu các nguồn còn lại.
3. Khởi động kernel với thư mục làm việc là gốc dự án. Nếu ứng dụng khởi động kernel ở thư mục khác, đặt `CV_PROJECT_ROOT` trỏ đến đường dẫn tuyệt đối của `DoAn_CV_NangCao` trước khi khởi động kernel. Chạy từng notebook bằng **Restart Kernel → Run All** theo thứ tự dưới đây.

| Thứ tự | Notebook | Tệp dùng ở bước sau |
|---|---|---|
| 1 | [Preprocessing validation](../preprocessing/Preprocessing_thi_diem_validation.ipynb), khi cần tạo lại NPZ | `preprocessing/notebook_outputs/pilot_prepared_pixels.npz` |
| 2 | [B0 validation](../danh_gia_b0/B0_evaluator_validation.ipynb), khi cần tạo lại embedding B0 | `danh_gia_b0/notebook_outputs/cache/embeddings.npz` và metadata |
| 3 | [Augmentation](../augmentation/Augmentation_che_co_kiem_soat.ipynb) | Manifest, đặc tả và bảng phân lớp trong `augmentation/notebook_outputs/` |
| 4 | [Pilot B1/B2](Pilot_train_B1_B2.ipynb) | Pixel train dạng `notebook_outputs/cache/train.npy`, predictions, checkpoint và báo cáo |
| 5 | [Phân tích và kiểm chứng](Phan_tich_va_kiem_chung_pilot.ipynb) | Bảng theo kiểu che và báo cáo kiểm resume trong `analysis_outputs/` |

Bước 2 đọc pixel NPZ từ bước 1. Bước 4 đọc embedding B0 và manifest augmentation, rồi tự dựng pixel cache khi thiếu. Bước 5 đọc predictions, checkpoint resume được chọn và ảnh DigiFace gốc. Với các tệp pilot đã có trong gói, có thể chạy riêng notebook phân tích sau khi đặt dữ liệu gốc.

Gói gồm notebook đã lưu output, bảng kết quả, encoder tốt nhất và checkpoint resume epoch 2 của mỗi nhánh. Cache và checkpoint resume epoch 0–1 được tạo lại khi chạy pipeline. Lượt pilot mới ghi vào cùng các đường dẫn output và checkpoint. Giữ bản sao kết quả cần đối chiếu trước khi chạy lượt mới.

## Checkpoint và số liệu gốc

- `checkpoints/B1_best_encoder.pt` và `checkpoints/B2_best_encoder.pt` chứa encoder ở epoch 2 để trích embedding. Nạp `encoder_state` vào `InceptionResnetV1(pretrained=None, classify=False)`.
- `checkpoints/B1_epoch2_resume.pt` và `checkpoints/B2_epoch2_resume.pt` chứa thêm classifier, optimizer, RNG và số bước để tiếp tục train. Bản làm việc còn có checkpoint resume của epoch 0 và 1.
- `notebook_outputs/tables/09_pilot_model_comparison.csv` lưu số liệu chính. `09_paired_identity_bootstrap_CI.csv` lưu khoảng tin cậy và `09_predictions.csv` lưu score từng query.
- `notebook_outputs/execution_summary.json` ghi phạm vi, cấu hình, môi trường và SHA-256 của encoder. `analysis_outputs/execution_summary.json` ghi kết quả phân tích và phép thử resume. Hai báo cáo đều ghi trạng thái `PASS`.
- Phép thử resume khớp loss và trọng số giữa hai lượt ở cùng bước sau lưu/nạp. Các ca thử chủ động bỏ optimizer hoặc RNG tạo ra sai lệch như dự kiến.
