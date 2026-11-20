# Chuỗi cải tiến B0 → B3

Cập nhật 20/11/2026. Dựa trên bài báo của Anwar & Raychowdhury (2020) [1]. Bài này dùng FaceNet [2], đeo khẩu trang giả lên ảnh mặt thật bằng công cụ MaskTheFace rồi train trên ảnh đó, và cũng là bài công bố bộ MFR2 mà nhóm dùng để test. Mỗi bước thêm đúng một thay đổi:

| | Thay đổi | Cơ sở |
|---|---|---|
| **B0** | FaceNet (InceptionResnetV1 pretrained VGGFace2 [3]) + MTCNN [4]; ngưỡng chọn trên ảnh thật rõ mặt (LFW [7]) | Baseline của [1] |
| **B1** | **Đổi trọng số:** fine-tune nhẹ trên 3.734 ảnh mặt thật LFW (406 người), 50% ảnh đeo khẩu trang/kính giả | Cách làm của [1] |
| **B2** | **Đổi ngưỡng và cách so khớp:** mỗi người đăng ký thêm template đeo khẩu trang giả và kính giả; ngưỡng chọn trên ảnh có che | Bước 7: ngưỡng phải chọn đúng miền ảnh |
| **B3** | **Thêm model song song:** điểm = 0,25 · FaceNet (B2) + 0,75 · ArcFace [5] | ArcFace R50 của InsightFace, train trên WebFace600K [6] |

B1/B2 cũ (fine-tune trên ảnh tổng hợp DigiFace [9], của Bảo và Trí) là thí nghiệm phụ. Thí nghiệm đó cho thấy train trên mặt tổng hợp làm model kém trên ảnh thật, xem `danh_gia_test/` và `model_tri/`.

## Kết quả trên ảnh che thật

Ngưỡng và trọng số ghép được chọn hoàn toàn trên dev LFW (174 người quen, 600 người lạ, không trùng ROF/MFR2), khóa trong `khoa_cau_hinh_B0_B3.json` trước khi chấm.

| | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|
| ROF khẩu trang, DIR | 27,0% | 39,3% | 53,9% | **83,1%** |
| ROF kính râm, DIR | 48,8% | 43,2% | 44,4% | **89,9%** |
| ROF rõ mặt, DIR | 92,0% | 92,0% | 86,1% | **97,2%** |
| ROF mọi query, DIR | 74,9% | 73,6% | 70,9% | **94,2%** |
| ROF mọi query, FPIR | 3,7% | 6,9% | 3,9% | **2,6%** |
| ROF mọi query, Rank-1 | 92,0% | 89,2% | 89,2% | **96,6%** |
| MFR2, độ chính xác tại ngưỡng khóa | 75,6% | 78,0% | 78,9% | **97,6%** |
| MFR2, AUC | 0,989 | 0,988 | 0,989 | **0,998** |

Hiệu từng bước trên ROF khẩu trang (DIR, CI 95% bootstrap theo người, 2.000 lần): B1 − B0 = +12,4 điểm [+3,7; +21,3]; B2 − B1 = +14,6 [+4,6; +26,2]; B3 − B2 = +29,2 [+20,5; +38,7].

**Đọc kết quả**

- **Trên khẩu trang thật, mục tiêu chính của đề tài, mỗi bước đều tốt hơn bước trước một cách có ý nghĩa.**
- **B1 và B2 có cái giá.**
  - B1 kém B0 trên kính râm thật.
  - B2 đặt ngưỡng cao hơn nên ảnh rõ mặt bị từ chối nhiều hơn.
  - Vì vậy, tính trên mọi query, B1 và B2 chưa hơn B0.
- **B3 hơn B0 ở mọi điều kiện, với FPIR thấp nhất.** Phần lớn sức mạnh của B3 đến từ ArcFace, vốn được train trên dữ liệu lớn hơn nhiều so với dữ liệu nhóm có.
- **Bài báo [1] train lại FaceNet từ đầu trên khoảng 700.000 ảnh** và báo TPR tăng khoảng 38%. Nhóm chỉ fine-tune trên vài nghìn ảnh, nên mức tăng của B1 nhỏ hơn nhưng cùng chiều.

## Notebook

| Notebook | Nội dung |
|---|---|
| `01_Du_lieu_LFW_va_che_gia.ipynb` | Kiểm kê LFW, loại 62 người trùng ROF/MFR2, tìm mặt, chia train/dev theo người, khẩu trang và kính giả |
| `02_Train_B1_LFW_khau_trang.ipynb` | Fine-tune B1 (mở `repeat_3`, `block8`, lớp chiếu; BatchNorm giữ nguyên thống kê), chọn epoch trên dev |
| `03_Danh_gia_B0_B3.ipynb` | Chọn ngưỡng/w trên dev, khóa, chấm ROF và MFR2, CI ghép cặp, hình cho slide |

Code dùng chung (tìm mặt, khẩu trang/kính giả, ArcFace, `LadderRecognizer` cho app demo) ở `cai_tien.py`.

Về file khóa:
- `khoa_cau_hinh_B0_B3_lan1.json` là lần khóa đầu.
- Lần khóa đó lưu SHA của file B1, và file này có kèm thời gian train nên đổi SHA mỗi lần train dù trọng số giống hệt.
- Sau khi bỏ thời gian train khỏi checkpoint, file khóa được ghi lại. Notebook 03 kiểm rằng ngưỡng và w trùng đúng lần đầu.

## Chạy lại

1. Tải dữ liệu ngoài (không có trên GitHub):
   ```powershell
   pip install pyarrow
   python ban_giao_nhom/cai_tien_B0_B3/tai_du_lieu_ngoai.py
   ```
   Lệnh tạo `data/external/lfw/images/` (13.233 ảnh) và `data/external/arcface/w600k_r50.onnx`.
2. Cần thêm ảnh ROF/MFR2 ở `data/benchmark_views/` và `data/raw/mfr2/`, như bước 7.
3. Chạy notebook 01 → 02 → 03 (Restart Kernel → Run All). Notebook 02 mất khoảng 2 phút trên RTX 4060; notebook 03 khoảng 8 phút, vì ArcFace chạy trên CPU.
4. Chỉ cần dùng model thì không phải chạy lại: `checkpoints/B1_LFW_mask_encoder.pt` có trong repo, app demo nạp qua `LadderRecognizer`.

## Tài liệu tham khảo

1. A. Anwar, A. Raychowdhury. *Masked Face Recognition for Secure Authentication*. arXiv:2008.11104, 2020. Mã nguồn MaskTheFace: https://github.com/aqeelanwar/MaskTheFace (MIT).
2. F. Schroff, D. Kalenichenko, J. Philbin. *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR 2015.
3. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. *VGGFace2: A Dataset for Recognising Faces across Pose and Age*. FG 2018.
4. K. Zhang, Z. Zhang, Z. Li, Y. Qiao. *Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks*. IEEE Signal Processing Letters, 2016.
5. J. Deng, J. Guo, N. Xue, S. Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR 2019. Model buffalo_l: https://github.com/deepinsight/insightface (chỉ dùng nghiên cứu phi thương mại).
6. Z. Zhu et al. *WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition*. CVPR 2021.
7. G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. UMass Amherst Technical Report 07-49, 2007.
8. M. E. Erakın, U. Demir, H. K. Ekenel. *On Recognizing Occluded Faces in the Wild*. BIOSIG 2021 (bộ ROF).
9. G. Bae et al. *DigiFace-1M: 1 Million Digital Face Images for Face Recognition*. WACV 2023.
10. P. Grother, M. Ngan, K. Hanaoka. *Face Recognition Vendor Test (FRVT) Part 2: Identification*. NISTIR 8271, 2019 (định nghĩa DIR/FPIR).
11. T. Esler. *facenet-pytorch*. https://github.com/timesler/facenet-pytorch
