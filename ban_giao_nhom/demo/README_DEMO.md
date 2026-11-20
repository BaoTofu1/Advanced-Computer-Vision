# Bước 8 — Demo webcam, thu ảnh và đo tốc độ

Cập nhật 16/11/2026. Thư mục này có ba file, dùng chung một pipeline (MTCNN tìm mặt → cắt 160×160 → InceptionResnetV1 → so cosine với gallery):

| File | Việc |
|---|---|
| `nhan_dien.py` | Pipeline dùng chung: `FacePipeline` (tìm mặt, trích embedding, đổi model B0/B1/B2), `Gallery` (đăng ký, nhận diện) và `Viewer` (cửa sổ hiển thị) |
| `thu_anh_webcam.py` | Thu ảnh webcam đúng cấu trúc thư mục cho ba phiên |
| `webcam_demo.py` | App demo: hiện tên / Unknown / không thấy mặt, cosine, ảnh đăng ký |

Tất cả lệnh chạy từ thư mục dự án `DoAn_CV_NangCao`, bằng Python của môi trường dự án (cài theo [README_GOI_BAN_GIAO.md](../README_GOI_BAN_GIAO.md)); không cần cài thêm gì. Môi trường dùng `opencv-python-headless` (đọc camera được nhưng không có `cv2.imshow`), nên cửa sổ hiển thị dùng tkinter có sẵn trong Python. Máy không có GPU vẫn chạy được, khoảng 16 khung/giây phần xử lý ([kết quả đo](../do_toc_do/Do_toc_do_pipeline.ipynb)).

## 1. Thu ảnh

Đọc [HUONG_DAN_THU_ANH_WEBCAM.md](../HUONG_DAN_THU_ANH_WEBCAM.md) trước (ai, bao nhiêu, ba phiên, xin đồng ý). Mỗi lần chạy thu một người, một điều kiện:

```powershell
python ban_giao_nhom/demo/thu_anh_webcam.py --phien s1_dang_ky    --ma-nguoi W01 --dieu-kien ro
python ban_giao_nhom/demo/thu_anh_webcam.py --phien s2_hieu_chinh --ma-nguoi W01 --dieu-kien khau_trang
python ban_giao_nhom/demo/thu_anh_webcam.py --phien s3_test       --ma-nguoi U06 --dieu-kien kinh
```

Space để chụp, `q` để kết thúc. Ảnh lưu vào `data/raw/webcam/<phiên>/<mã người>/<điều kiện>_<số>.jpg`, mỗi lần chụp ghi một dòng vào `data/raw/webcam/nhat_ky_thu_anh.csv`. Phiên đăng ký chỉ lưu khi thấy đúng một mặt; phiên 2 và 3 lưu cả ảnh detector không thấy mặt, vì đó là lỗi thật cần đo.

Điều kiện: `ro`, `khau_trang`, `kinh`, `tay_che`, `nghieng`, `toi`. Người đăng ký dùng mã `W01, W02…`; người lạ dùng `U01, U02…` và **không** chụp phiên đăng ký. Người lạ của phiên 2 và phiên 3 phải là hai nhóm khác nhau. Tự điền `data/raw/webcam/nguoi_tham_gia.csv`:

```text
ma_nguoi,vai_tro,phien_tham_gia,dong_y,ngay_xoa_du_kien
W01,known,s1;s2;s3,yes,2027-01-31
U01,unknown,s2,yes,2027-01-31
U06,unknown,s3,yes,2027-01-31
```

## 2. Chọn ngưỡng và test

Sau khi thu đủ ba phiên, mở [../webcam/Hieu_chinh_va_test_webcam.ipynb](../webcam/Hieu_chinh_va_test_webcam.ipynb) → Restart Kernel → Run All. Notebook kiểm cấu trúc dữ liệu, chọn ngưỡng cho B0/B1/B2 trên phiên 2 (FPIR ≤ 5%), khóa ngưỡng, chấm phiên 3 theo từng điều kiện, rồi ghi `demo/nguong_webcam.json` cho app. Chưa đủ dữ liệu thì notebook liệt kê còn thiếu gì.

## 3. Chạy demo

```powershell
python ban_giao_nhom/demo/webcam_demo.py                      # webcam 0, model B0, gallery = data/raw/webcam/s1_dang_ky
python ban_giao_nhom/demo/webcam_demo.py --model B2           # thử model khác
python ban_giao_nhom/demo/webcam_demo.py --threshold 0.55     # đặt ngưỡng tay
python ban_giao_nhom/demo/webcam_demo.py --device cpu         # ép chạy CPU
```

| Phím | Tác dụng |
|---|---|
| `q` / Esc | Thoát |
| `1` / `2` / `3` | Đổi model B0 / B1 / B2 |
| `+` / `-` | Tăng / giảm ngưỡng 0,01 |
| `r` | Nạp lại gallery từ thư mục (sau khi đăng ký thêm người) |

Màn hình hiện: khung mặt và tên (xanh) hoặc `Unknown` (cam) kèm cosine; "Không tìm thấy mặt" / "Nhiều mặt trong khung hình" (đỏ); ảnh đăng ký của người được nhận; dòng dưới cùng ghi model, ngưỡng và nguồn của ngưỡng, số người trong gallery, thời gian xử lý, FPS.

**Ngưỡng.** App đọc `nguong_webcam.json` nếu notebook webcam đã tạo. Chưa có thì dùng tạm ngưỡng chọn trên DigiFace (B0: 0,693), dòng trạng thái ghi "tạm, từ DigiFace". Ngưỡng tạm này khá chặt với ảnh thật: ít nhận nhầm người lạ nhưng hay trả Unknown cho người quen. Với B1/B2 thì ngưỡng DigiFace quá lỏng trên ảnh thật (bước 7: nhận nhầm khoảng 90% người lạ), nên đừng demo B1/B2 bằng ngưỡng tạm.

**Model.** Bước 7 cho thấy B0 tốt nhất trên ảnh người thật, nên mặc định là B0. Phím `2`, `3` để cho người xem thấy khác biệt với B1/B2.

**Chạy trên ảnh hoặc video có sẵn** (không cần webcam; dùng để thử hoặc chấm lô):

```powershell
python ban_giao_nhom/demo/webcam_demo.py --source thu_muc_anh --no-window --out ket_qua --save-frames
```

Ghi `ket_qua/predictions.csv` (mỗi khung một dòng: trạng thái, tên dự đoán, tên gần nhất, cosine, ngưỡng, model, trạng thái detector, thời gian xử lý) và khung hình đã vẽ.

## 4. Giới hạn cần nói khi demo

- Mỗi khung hình nhận diện độc lập, không làm mượt theo thời gian. Mất mặt thì nhãn mất ngay; app không giữ tên từ khung trước.
- Khung hình phải có đúng một người (phạm vi đề tài). Nhiều mặt thì app báo, không chọn một mặt để nhận.
- Cosine là độ giống, không phải xác suất đúng.
- Không có chống giả mạo: ảnh in hay màn hình điện thoại vẫn có thể được nhận. Không dùng để mở cửa hay xác thực.
- MTCNN hay bỏ sót mặt đeo khẩu trang (bước 7: chỉ thấy 72,7% ảnh khẩu trang của ROF). Khi demo khẩu trang, "Không tìm thấy mặt" là kết quả dự kiến ở một phần khung hình.

## 5. Đã kiểm gì

- App demo đã chạy trên thư mục ảnh và trên file video (B0 và B2), ghi đúng `predictions.csv`; lớp vẽ đã xem trên khung 640×480.
- Cửa sổ tkinter đã chạy ở chế độ ẩn với phím giả lập (đổi model, chỉnh ngưỡng, chụp, thoát đều được nhận).
- Notebook webcam đã chạy trọn trên một bộ ảnh diễn tập (ảnh ROF xếp theo cấu trúc thư mục webcam) để kiểm code. Kết quả diễn tập không được lưu và không dùng trong báo cáo.

**Chưa kiểm:** chạy với camera thật và cửa sổ hiện trên màn hình (cần người ngồi trước máy); `thu_anh_webcam.py` chưa chụp ảnh thật lần nào. Lần chạy đầu với webcam nên thử hết các phím trước khi mời người tham gia.
