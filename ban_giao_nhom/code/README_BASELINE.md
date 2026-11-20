# Baseline B0: so sánh ảnh rõ và ảnh che khuất

`face_occlusion_baseline_compare.py` nhận một ảnh tham chiếu và nhiều ảnh query, chạy MTCNN → crop 160 × 160 → InceptionResnetV1 pretrained VGGFace2 → embedding 512 chiều chuẩn hóa L2 → cosine similarity. Đây là bước kiểm tra B0 và luồng xử lý; chưa phải bộ đánh giá 1:N, chương trình webcam hoặc code fine-tune B1/B2.

## 1. Cài đặt trong môi trường riêng

Khuyến nghị Python 3.10 hoặc 3.11, Windows 64-bit. Các bản pin lấy theo giới hạn phụ thuộc của `facenet-pytorch==2.6.0`: torch 2.2.x, torchvision 0.17.x, NumPy < 2 và Pillow 10.2.x. Không dùng chung môi trường EDA nếu môi trường đó đang có NumPy 2.

Mở PowerShell tại thư mục `code` của gói bàn giao. Lệnh sau dùng CPU để dễ tái lập và không cần CUDA:

```powershell
py -3.11 -m venv .venv-baseline
.\.venv-baseline\Scripts\python.exe -m pip install --upgrade pip
.\.venv-baseline\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
.\.venv-baseline\Scripts\python.exe -m pip install -r requirements-baseline.txt
.\.venv-baseline\Scripts\python.exe -m pip check
.\.venv-baseline\Scripts\python.exe face_occlusion_baseline_compare.py --help
```

Nếu dùng GPU, chọn wheel CUDA cho đúng máy từ trang cài đặt PyTorch và giữ đúng cặp `torch==2.2.2`, `torchvision==0.17.2`. Không cài lại các gói tùy tiện vào môi trường đã chốt. Ghi `pip freeze` và tên GPU vào biên bản thí nghiệm.

Lần đầu khởi tạo encoder sẽ tải trọng số VGGFace2 khoảng 107 MB qua GitHub; cần mạng. MTCNN có các trọng số detector trong gói. `--weights-dir` đặt TORCH_HOME, và encoder cache tại `checkpoints/20180402-114759-vggface2.pt` bên trong thư mục đó. Đây là cache trọng số, không phải thư mục chứa dataset VGGFace2.

## 2. Chạy thử

Thay các đường dẫn ví dụ bằng ảnh thật. Từng ảnh phải chứa đúng một mặt được MTCNN phát hiện. Có thể dùng cùng một ảnh ở hai phía để kiểm tra cosine gần 1 trước, rồi so sánh ảnh rõ, ảnh khẩu trang và ảnh kính của cùng người; thêm ảnh người khác để kiểm tra tương phản.

```powershell
.\.venv-baseline\Scripts\python.exe .\face_occlusion_baseline_compare.py `
  --reference "..\data_demo\anh_ro.jpg" `
  --queries "..\data_demo\anh_ro.jpg" "..\data_demo\khau_trang.jpg" "..\data_demo\nguoi_khac.jpg" `
  --output "..\outputs\baseline_compare.json" `
  --device cpu `
  --weights-dir "..\model_cache"
```

Đường dẫn có khoảng trắng hoặc tiếng Việt phải đặt trong dấu nháy kép. Script đọc qua pathlib và Pillow, áp dụng hướng EXIF rồi đổi RGB ở cả reference/query; không đưa ảnh BGR của OpenCV vào trực tiếp. Nếu ảnh có nhiều người, script báo `multiple_faces` thay vì tự chọn mặt to nhất. Không có mặt được phát hiện thì báo `no_face`. **Từ bản 1.0.1**, ảnh có cạnh nhỏ hơn `min_face_size=20` trả `image_too_small` trước khi gọi detector, giữ `face_count=null` vì chưa phát hiện mặt; không phóng ảnh tự động để né lỗi. Ảnh che nặng vẫn có thể lỗi detector; phải giữ các lỗi trong kết quả đánh giá hệ thống.

Crop hiện tại là crop theo bounding box MTCNN, resize và chuẩn hóa `(x - 127.5) / 128`. **Chưa có phép alignment theo landmark 5 điểm**; không mô tả nhầm baseline này là đã alignment hình học. Đây là một quyết định cần kiểm tra bằng EDA detector và thí nghiệm trước khi viết preprocessing chính thức.

## 3. Đọc kết quả JSON (schema 1.0)

| Trường | Ý nghĩa |
|---|---|
| `environment` | Phiên bản Python, hệ điều hành, các package, CUDA và GPU nếu khởi tạo được |
| `config` | Detector, kích thước crop, chuẩn hóa, pretrained, loại score; `identity_threshold` luôn null |
| `reference` | Đường dẫn tuyệt đối, SHA-256, kích thước sau EXIF, số mặt, boxes, detector scores, trạng thái và lỗi |
| `queries[]` | Cùng metadata như reference, thêm `cosine_similarity`, `comparison_error` |
| `summary` | Tổng query, số embedding thành công, số có score, số lỗi và exit code |
| `fatal_error` | Lỗi môi trường, tải trọng số hoặc khởi tạo model; null nếu không có |

`cosine_similarity` nằm trong [-1, 1], càng lớn thì embedding càng giống nhau. **Không phải xác suất cùng người**, không nhân 100 để gọi là “độ chính xác”. `detector_scores` chỉ mô tả độ tin cậy phát hiện mặt. Không có kết luận same/different/Unknown trong script này vì chưa hiệu chỉnh threshold trên validation.

`status` của mỗi ảnh là `ok`, `error` hoặc `not_processed`. Query có embedding `ok` vẫn có thể không so sánh được khi reference lỗi; lúc đó `comparison_error.code=reference_invalid`, score null. Lỗi môi trường vẫn tạo JSON với trạng thái `not_processed` cho các ảnh chưa chạy. File output được ghi nguyên tử và ghi đè nếu tồn tại; script từ chối output trùng đường dẫn input. Output không lưu embedding hoặc ảnh crop.

Exit code: **0** khi mọi so sánh thành công; **1** khi có query lỗi; **2** khi reference không hợp lệ, lỗi môi trường, CUDA hoặc lỗi lưu file. Lỗi argparse cũng trả 2. Đọc `$LASTEXITCODE` ngay sau lệnh chạy nếu dùng PowerShell.

## 4. Giới hạn và việc cần làm trước báo cáo accuracy

- Một vài cặp ảnh chỉ chứng minh code chạy; không đo được độ chính xác hoặc hiệu quả che khuất của hệ thống.
- Để so B0/B1/B2 phải có protocol gallery/query/unknown cố định, danh tính tách biệt với fine-tune, threshold chọn trên validation và khóa trước test. Không dùng kết quả test để sửa cấu hình.
- Trọng số B0 đã pretrained trên VGGFace2. Nếu fine-tune/evaluate trên dữ liệu người nổi tiếng, cần kiểm tra chồng lặp với VGGFace2/CASIA theo khả năng truy vết. “Không xuất hiện trong fine-tune của nhóm” không đồng nghĩa “chưa xuất hiện trong pretraining”. Ghi rõ giới hạn nếu không loại trừ được.
- Tách báo cáo recognition trên crop hợp lệ và end-to-end có tính detection failure. Không loại query lỗi rồi gọi phần còn lại là accuracy toàn hệ thống.
- Chưa train B1/B2, chưa lựa chọn augmentation/quality cutoff và chưa preprocessing dataset hàng loạt trong script này.

## 5. Kiểm chứng bản bàn giao

Đã kiểm tra cú pháp, `--help`, luồng thiếu dependency ghi JSON/exit 2 và cơ chế từ chối output trùng input bằng Python 3.12 bundled. Đã dùng detector giả để kiểm tra nhánh 0 mặt, nhiều mặt, ảnh không đọc được và ảnh grayscale được đổi RGB; kiểm tra này không đánh giá detector hoặc encoder thật. Đã sửa lỗi console Windows khi in đường dẫn tiếng Việt.

**Đã chạy inference thật ngày 21/10/2026** bằng Python 3.11.9, torch 2.2.2+cu121, torchvision 0.17.2+cu121, facenet-pytorch 2.6.0, NumPy 1.26.4, Pillow 10.2.0; GPU NVIDIA GeForce RTX 4060 Laptop GPU. Output: `validation/b0_smoke_real.json`. Reference `BenAffleck_0003.png` có nhãn nguồn no-mask. Kết quả smoke test:

| Query theo metadata MFR2 | Cosine | Kết quả kỹ thuật |
|---|---:|---|
| Cùng ảnh reference | 1,000000 | PASS, một mặt |
| `BenAffleck_0007.png`, cùng ID, no-mask | 0,815238 | PASS, một mặt |
| `BenAffleck_0004.png`, cùng ID, surgical_black | 0,476468 | PASS, một mặt |
| `AhmadAli_0002.png`, khác ID, no-mask | -0,042877 | PASS, một mặt |
| File cố ý không tồn tại | null | PASS, `image_read_failed`, toàn lệnh exit 1 |

Các cặp này được chọn theo nhãn nguồn để xác minh luồng code; **không phải đánh giá benchmark**, không dùng suy ra accuracy, hiệu quả model hoặc chọn threshold. Không có ảnh raw nào được chép vào thư mục code bàn giao. Chưa đo latency chuẩn hóa. File `validation/missing_dependencies.json` là bằng chứng thử luồng lỗi trên môi trường bundled trước khi cài môi trường ML, không mô tả môi trường GPU hiện tại. `validation/validation_summary.json` ghi trạng thái kiểm chứng cuối.

Bản **1.0.1** bổ sung kiểm tra ảnh nhỏ sau khi EDA phát hiện crop ROF có cạnh 15–19 pixel khiến MTCNN không dựng được image pyramid. Đã chạy thêm kiểm thử CUDA với ảnh RGB 16 × 22 sinh tạm: query trả `image_too_small`, `face_count=null`, không có cosine và toàn lệnh exit 1 đúng dự kiến; lưu `validation/tiny_image_guard.json`. Ảnh thử đã được xóa cùng thư mục tạm, không thêm vào dataset. Đã chạy lại smoke test 4 query hợp lệ và file thiếu bằng bản 1.0.1. Các log lỗi detector của EDA trước bản sửa vẫn được giữ làm bằng chứng, không viết lại kết quả cũ.

## Nguồn chính thức

- Package metadata 2.6.0: https://pypi.org/pypi/facenet-pytorch/2.6.0/json
- MTCNN và chuẩn hóa: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py
- Encoder, trọng số và cache: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py
- PyTorch phiên bản trước: https://pytorch.org/get-started/previous-versions/
