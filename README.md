# Đồ án CV nâng cao: Nhận diện khuôn mặt khi bị che khuất

Ứng dụng webcam nhận diện người đã đăng ký khi đeo khẩu trang, kính râm hoặc bị che một phần mặt; trả về **Unknown** khi không đủ bằng chứng. Nhánh chính: MTCNN + InceptionResnetV1 (facenet-pytorch). So sánh **B0** (pretrained), **B1** (fine-tune trên dữ liệu nền) và **B2** (fine-tune trên dữ liệu nền có thêm che nhân tạo).

## Tiến độ (cập nhật 20/11/2026)

**Kết quả chính (20/11/2026): chuỗi cải tiến B0 → B3** dựa trên bài báo Anwar & Raychowdhury 2020 (MaskTheFace/MFR2), xem [README_CAI_TIEN](ban_giao_nhom/cai_tien_B0_B3/README_CAI_TIEN.md). Trên khẩu trang thật (ROF), tỷ lệ nhận đúng người quen tăng qua từng bước: **B0 27,0% → B1 39,3% → B2 53,9% → B3 83,1%**; trên MFR2 độ chính xác 75,6% → 78,0% → 78,9% → **97,6%**. App demo mặc định dùng B3 và đổi được B0–B3 bằng phím 1–4.

B1/B2 cũ (fine-tune trên ảnh tổng hợp DigiFace, lượt của Bảo và của Trí) giữ lại như thí nghiệm phụ: tốt trên DigiFace nhưng kém B0 trên ảnh thật (bước 7).

**Việc còn lại là của người, không phải của code:** thu ảnh webcam ba phiên, chạy notebook webcam, chạy thử demo trước camera, làm báo cáo/slide.

| Bước | Trạng thái |
|---|---|
| 01. Thu thập dữ liệu + EDA | ✅ Xong: `ban_giao_nhom/eda/EDA_kham_pha_du_lieu.ipynb` |
| 02. QA nhãn / ảnh trùng / bounding box | ✅ Xong phần đề xuất: `ban_giao_nhom/qa/QA_nhan_trung_bounding_box.ipynb`. **Chờ nhóm duyệt** |
| 03. Ánh xạ danh tính, split, gallery/query/unknown | ✅ Xong: `ban_giao_nhom/protocol/Chia_tap_va_protocol_danh_gia.ipynb` (DigiFace + ROF/MFR2). 5 chính sách P1–P5 **đã được xác nhận** (28/10): `ban_giao_nhom/quyet_dinh_nhom/` |
| 04. Preprocessing thí điểm | ✅ Xong pilot trên validation: `ban_giao_nhom/preprocessing/Preprocessing_thi_diem_validation.ipynb` |
| 05. B0 + evaluator 1:N trên validation | ✅ Xong: `ban_giao_nhom/danh_gia_b0/B0_evaluator_validation.ipynb`. DIR tại FPIR 5% = **57,2%** (cách A) |
| 06. Sinh che nhân tạo, train B1/B2 | ✅ Xong: augmentation, pilot 128 người (Codex) và **train full 1.600 người**: `ban_giao_nhom/huan_luyen/Train_full_B1_B2.ipynb`. DIR ảnh nền B1 = B2 = **93,5%**; DIR query che B1 56,5%, **B2 79,6%** |
| 07. Khóa cấu hình/ngưỡng, test độc lập | ✅ Xong: `ban_giao_nhom/danh_gia_test/Danh_gia_doc_lap_B0_B1_B2.ipynb`. ROF Rank-1 B0 **92,8%**, B1 72,2%, B2 67,2%; MFR2 AUC B0 **0,991**, B1 0,935, B2 0,944 |
| 07b. Chuỗi cải tiến B0 → B3 | ✅ Xong: `ban_giao_nhom/cai_tien_B0_B3/` (LFW + khẩu trang giả, B1 fine-tune, B2 template che, B3 + ArcFace). ROF khẩu trang DIR **27% → 39% → 54% → 83%** |
| 08a. Đo tốc độ | ✅ Xong: `ban_giao_nhom/do_toc_do/Do_toc_do_pipeline.ipynb`. Một khung 640×480 mất ~57 ms trên GPU, ~64 ms trên CPU (16–17 khung/giây); 2/3 thời gian là detector |
| 08b. App demo + công cụ thu ảnh | ✅ Code xong, đã kiểm trên ảnh/video: `ban_giao_nhom/demo/`. ⏳ Chưa chạy thử trước camera thật |
| 08c. Ngưỡng và test webcam | ⏳ Notebook viết sẵn (`ban_giao_nhom/webcam/Hieu_chinh_va_test_webcam.ipynb`), **chờ nhóm thu ảnh** |
| 08d. Báo cáo, slide | ⏳ Chưa làm |

## Branch trên GitHub

Mỗi branch là một mốc của dự án, branch sau chứa toàn bộ branch trước. `main` là bản đầy đủ.

| Branch | Thêm gì so với branch trước |
|---|---|
| `Data` | Nguồn dữ liệu, log tải DigiFace, annotation ROF, metadata MFR2, hướng dẫn triển khai, script baseline. Không có ảnh |
| `EDA` | Notebook EDA và QA (nhãn, ảnh trùng, bounding box) |
| `Preprocess` | Chia tập + protocol, quyết định nhóm P1–P5, preprocessing thí điểm |
| `Model` | Evaluator 1:N và B0, sinh che nhân tạo, pilot B1/B2 (phần chung của hai lượt train) |
| `Model-Bao` | Train full B1/B2 của Bảo (notebook, model đã lưu, `tai_model.py`) và đánh giá độc lập bước 7 |
| `Model-Tri` | Train B1/B2 của Trí (script, log, model) và notebook so sánh hai lượt train |
| `Model-B0-B3` | Chuỗi cải tiến B0 → B3 theo bài báo Anwar & Raychowdhury 2020: dữ liệu LFW, train B1, chấm ROF/MFR2 (nối tiếp `Model`) |
| `Web-Demo` | App demo webcam (B0–B3), công cụ thu ảnh, đo tốc độ, notebook ngưỡng + test webcam (gộp `Model-Bao` và `Model-B0-B3`) |
| `main` | Gộp tất cả |

Không có trên GitHub: ảnh gốc (`data/raw`, `data/benchmark_views`), LFW và model ArcFace (`data/external`, tải bằng `cai_tien_B0_B3/tai_du_lieu_ngoai.py`), ảnh webcam, cache, checkpoint để train tiếp, trang ảnh khuôn mặt để duyệt tay (contact sheets). Hai model B1/B2 của mỗi lượt train (~90 MB/file) có trong repo.

## Đọc theo thứ tự này

1. **File này**: tổng quan và các quyết định cần chốt.
2. `ban_giao_nhom/huong_dan_trien_khai_hoan_chinh.pdf` (16 trang): kế hoạch, protocol đánh giá, phân công A/B/C, mốc thời gian.
3. `ban_giao_nhom/eda/EDA_kham_pha_du_lieu.ipynb`: dữ liệu thực tế có gì, vấn đề gì; phần 23 là kết luận.
4. `ban_giao_nhom/qa/QA_nhan_trung_bounding_box.ipynb`: kết quả kiểm nhãn/ảnh trùng/box; phần 20 là bảng quyết định, phần 22 là kết quả.
5. `ban_giao_nhom/protocol/Chia_tap_va_protocol_danh_gia.ipynb`: chính sách draft, split theo người, known/unknown, gallery/query, kiểm rò rỉ và feasibility ảnh thật.
6. `ban_giao_nhom/quyet_dinh_nhom/Cac_quyet_dinh_can_xac_nhan.ipynb`: 5 chính sách P1–P5 (đã xác nhận) và 58 trang ảnh gallery để duyệt.
7. `ban_giao_nhom/preprocessing/Preprocessing_thi_diem_validation.ipynb`: so hai cách chuẩn bị ảnh trên validation.
8. `ban_giao_nhom/danh_gia_b0/B0_evaluator_validation.ipynb`: evaluator 1:N có người lạ và kết quả B0 (mốc để so B1/B2).
9. [Hướng dẫn và kết quả pilot bước 6](ban_giao_nhom/huan_luyen/README_PILOT.md), gồm thứ tự chạy lại và checkpoint.
10. [Notebook augmentation](ban_giao_nhom/augmentation/Augmentation_che_co_kiem_soat.ipynb), [notebook train B1/B2](ban_giao_nhom/huan_luyen/Pilot_train_B1_B2.ipynb) và [notebook kiểm chứng pilot](ban_giao_nhom/huan_luyen/Phan_tich_va_kiem_chung_pilot.ipynb).
11. [Kết quả train full và model đã lưu](ban_giao_nhom/huan_luyen/README_TRAIN_FULL.md), [notebook train full B1/B2](ban_giao_nhom/huan_luyen/Train_full_B1_B2.ipynb). Kết quả chính thức của bước 6 nằm ở đây, không phải ở pilot.
12. [Kết quả bước 7](ban_giao_nhom/danh_gia_test/README_DANH_GIA_TEST.md) và [notebook đánh giá độc lập](ban_giao_nhom/danh_gia_test/Danh_gia_doc_lap_B0_B1_B2.ipynb): DigiFace test, ROF, MFR2.
13. [Model B1/B2 của Trí và so sánh hai lượt train](ban_giao_nhom/model_tri/README_MODEL_TRI.md): cùng evaluator, cùng kết luận (tốt trên DigiFace, kém B0 trên ảnh thật).
14. **[Chuỗi cải tiến B0 → B3](ban_giao_nhom/cai_tien_B0_B3/README_CAI_TIEN.md)**: kết quả chính, kèm tài liệu tham khảo.
15. Bước 8: [hướng dẫn demo và thu ảnh](ban_giao_nhom/demo/README_DEMO.md), `ban_giao_nhom/HUONG_DAN_THU_ANH_WEBCAM.md` (ai, bao nhiêu, ba phiên), [notebook đo tốc độ](ban_giao_nhom/do_toc_do/Do_toc_do_pipeline.ipynb), [notebook ngưỡng + test webcam](ban_giao_nhom/webcam/Hieu_chinh_va_test_webcam.ipynb).

Notebook đã lưu sẵn output, nên mở ra đọc được ngay mà không cần chạy lại hay tải dữ liệu.

## Phần tiếp nối đã thực hiện

- Sửa luồng duyệt QA: `ban_giao_nhom/qa/group_decisions.csv` là input riêng, không bị Run All ghi đè; notebook kiểm đầy đủ ID/người duyệt/lý do/ngày/bằng chứng rồi merge vào báo cáo. Tệp thật hiện chưa có quyết định của thành viên.
- Split draft DigiFace: 1.600 / 200 / 200 danh tính train/validation/test. Train có 57.600 ảnh train và 6.400 monitor. Mỗi val/test có 100 known và 100 unknown: 300 ảnh gallery nền, 3.700 known-query, 4.000 unknown-query.
- Chính sách draft giữ 86.094/86.980 ảnh, cách ly 886 hold/secondary trên danh sách. Giữ cả 2.946 ảnh cần review kèm cờ; không loại ca detector/quality chỉ để làm đẹp kết quả.
- ROF/MFR2 dành ngoài tuning. Chưa có gallery external chính thức hoặc ngưỡng nhận diện; alias vẫn là guard, không tự gộp người.

- Preprocessing thí điểm dùng 308 ảnh DigiFace validation, gồm 300 ảnh mẫu chính và 8 ảnh detector từng thất bại ở EDA. **A – resize thẳng** chuẩn bị được 308/308 ảnh. **B – chạy lại MTCNN rồi crop** được 296/308, gồm 11 ca không thấy mặt và 1 ca nhiều mặt. Cả 604 tensor đạt kiểm tra chuẩn hóa `(pixel − 127,5)/128`. P5 dùng A cho DigiFace. B là ứng viên cho ảnh toàn cảnh và webcam.

- **B0 trên DigiFace validation** (8.000 ảnh, 100 người quen + 100 người lạ): ở mức người lạ bị nhận nhầm 5%, cách A nhận đúng **57,2%** query người quen (CI 95%: 52,8–61,6%), cách B **54,7%**. A hơn B khoảng 2,4 điểm, nghiêng về A nhưng bằng chứng còn yếu; kết quả ủng hộ P5. Rank-1 bỏ qua ngưỡng là 82,9%. Lỗi chủ yếu là người quen bị từ chối (41%) ở các ảnh nghiêng hẳn, tối hoặc bị che. Đây là mốc mà B1/B2 phải vượt.

Các con số trên là kết quả trên ảnh render DigiFace; chưa có kết quả trên ảnh che thật (ROF/MFR2, bước 7) hay webcam (bước 8). Gallery DigiFace có điều kiện `synthetic_unannotated`, chưa được gọi là ảnh rõ. Notebook lưu các điều kiện còn mở trước đánh giá xác nhận.

## Kết quả bước 6

### Train full (kết quả chính thức)

1.600 người train, 57.600 ảnh/epoch, 5 epoch, batch 8, fine-tune toàn bộ mạng, 36.000 bước mỗi nhánh (cấu hình giống lượt train song song của nhóm). Đánh giá trên DigiFace validation, mỗi model chọn ngưỡng trên ảnh nền ở FPIR ≤ 5%.

| Model | DIR ảnh nền | DIR query che (ngưỡng nền) | FPIR query che |
|---|---:|---:|---:|
| B0 | 57,16% | 22,73% | 2,75% |
| B1 (epoch 3) | 93,49% | 56,51% | 4,33% |
| B2 (epoch 4) | 93,49% | **79,62%** | 4,70% |

B2 ngang B1 trên ảnh nền (chênh 0,00 điểm, CI 95% [−1,00; +1,00]) và hơn B1 **23,1 điểm** trên query che (CI 95% [+20,8; +25,7]). Lợi lớn nhất ở che mắt (39,4% → 78,7%) và che nửa dưới (20,1% → 57,6%). So với lượt train song song (B1 94,78%, B2 94,54% trên ảnh nền), lệch khoảng 1 điểm và nằm trong CI. Model đã lưu ở `ban_giao_nhom/huan_luyen/checkpoints_full/`, nạp bằng `tai_model.py`; chi tiết ở [README train full](ban_giao_nhom/huan_luyen/README_TRAIN_FULL.md).

Đây vẫn là ảnh tổng hợp và che hình chữ nhật. Model fine-tune trên DigiFace có thể kém B0 trên ảnh người thật; bước 7 trên ROF/MFR2 sẽ trả lời.

### Pilot (Codex, kiểm code trước khi train full)

Pilot dùng 128 danh tính train, gồm 4.608 ảnh train và 512 ảnh monitor. B1 dùng ảnh nền DigiFace. B2 dùng cùng thứ tự ảnh và thêm che nhân tạo ở khoảng 50% lượt ảnh. Hai nhánh dùng cùng khởi tạo, 3 epoch và 432 bước optimizer mỗi nhánh. Tập DigiFace validation có 8.000 ảnh, trong đó 7.700 ảnh query được tạo thêm một biến thể che cố định để so sánh từng cặp.

| Model | DIR ảnh nền tại FPIR 5% | DIR ảnh che dùng ngưỡng ảnh nền | FPIR ảnh che |
|---|---:|---:|---:|
| B0 | 57,16% | 22,73% | 2,75% |
| B1 | 66,49% | 30,30% | 3,08% |
| B2 | 66,35% | 31,27% | 3,00% |

B2 cao hơn B1 **0,97 điểm phần trăm** DIR trên query che, với CI 95% bootstrap theo danh tính từ **+0,05 đến +1,97 điểm**. Chênh lệch trên ảnh nền là **−0,14 điểm**, CI từ **−1,22 đến +0,95 điểm**. Các khoảng này tính tại checkpoint và ngưỡng đã chọn từ validation. Pilot dùng một seed và ảnh che hình chữ nhật tổng hợp. Số liệu từng kiểu che và mẫu số nằm trong [notebook kiểm chứng](ban_giao_nhom/huan_luyen/Phan_tich_va_kiem_chung_pilot.ipynb).

Trên RTX 4060 Laptop, phần train và validation hai nhánh mất 88 giây. Đỉnh cấp phát tensor trong các epoch khoảng 362 MiB. Notebook đã kiểm gradient của encoder, thống kê BatchNorm, checkpoint encoder và việc resume có optimizer và RNG. Cách nạp checkpoint và chạy lại nằm trong [hướng dẫn pilot](ban_giao_nhom/huan_luyen/README_PILOT.md).

## Kết quả bước 7 (đánh giá độc lập)

Model, ngưỡng validation, tiền xử lý và cách chấm được khóa trong `ban_giao_nhom/danh_gia_test/khoa_cau_hinh_buoc7.json` trước khi mở ảnh test.

| | B0 | B1 | B2 |
|---|---:|---:|---:|
| DigiFace test, ảnh nền (DIR) | 62,9% | 95,4% | 95,2% |
| DigiFace test, che nhân tạo (DIR) | 25,5% | 59,0% | **81,1%** |
| ROF ảnh thật, mọi query (Rank-1) | **92,8%** | 72,2% | 67,2% |
| ROF khẩu trang thật (Rank-1) | **80,3%** | 34,8% | 36,5% |
| MFR2 (AUC) | **0,991** | 0,935 | 0,944 |

Trên ảnh tổng hợp, kết quả bước 6 giữ được trên 200 người mới. Trên ảnh người thật, B1/B2 kém B0 rõ rệt (ROF Rank-1: B2 − B0 = −25,6 điểm, CI 95% [−29,1; −22,3]): fine-tune toàn bộ mạng trên ảnh render làm model mất khả năng phân biệt mặt thật. Ngưỡng chọn trên DigiFace làm B1/B2 nhận nhầm khoảng 90% người lạ trên ảnh thật. Chi tiết và cách đọc ở [README bước 7](ban_giao_nhom/danh_gia_test/README_DANH_GIA_TEST.md).

## Dữ liệu hiện có

| Nguồn | Vai trò | Số lượng đã kiểm |
|---|---|---|
| DigiFace-1M (P1, ảnh tổng hợp) | Train | 80.000 ảnh / 2.000 người (40 ảnh/người, lấy từ archive 144.000 ảnh) |
| ROF (ảnh che thật, crawl theo URL) | Đánh giá chính | 6.711 crop / 182 nhãn, từ 6.296/9.520 URL còn tải được |
| MFR2 (khẩu trang thật) | Đánh giá phụ | 269 ảnh / 53 người, 848 cặp |
| Webcam của nhóm | Demo | Chưa thu |

CASIA-WebFace (kế hoạch ban đầu) không tải được vì link bị giới hạn lượt tải, nên phải dùng DigiFace thay thế.

## Phát hiện chính

- Cùng một crop ROF mang hai nhãn khác nhau (`kylian_mbappe` và `neymar`); 3 file nguồn là logo, không có mặt (34 crop bị ảnh hưởng); 723 crop thuộc ảnh có nhiều khuôn mặt nhưng chỉ gắn một tên.
- Detector MTCNN từ chối 30,30% crop ROF đeo khẩu trang, so với 6,13% ở ảnh rõ.
- DigiFace vốn đã có mũ, kính, bóng tối che mắt, nên B1 phải gọi là "dữ liệu nền chưa thêm che", không phải "ảnh rõ".
- QA đề xuất: **804 ảnh tạm giữ, 82 bản trùng phụ, 2.946 ảnh cần xem, 83.148 ảnh chưa phát hiện lỗi**. Chưa áp dụng vào dữ liệu.

## Nhóm cần làm ngay

1. **Điền tên** người phụ trách A/B/C (mục 12 của PDF) và deadline thực.
2. **Duyệt bảng quyết định QA** (notebook QA, phần 20) và ghi vào input riêng `qa/group_decisions.csv`, không sửa bảng output. Phần xem ảnh trước đây do trợ lý AI (Codex) thực hiện, **chưa có thành viên nào xác nhận**; riêng 723 crop từ entry nhiều box chưa xác định người chiếm phần lớn số ảnh tạm giữ.
3. ~~Chốt chính sách P1–P5~~: **đã xác nhận ngày 28/10** (cách ly 886 ảnh hold/trùng phụ; gallery DigiFace 3 ảnh nền/người; ROF/MFR2 chỉ để test; ngưỡng theo FPIR 5% trên validation; DigiFace dùng resize thẳng). Việc duyệt 1.032 ảnh gallery là tùy chọn, chỉ cần nếu muốn gọi gallery là "ảnh rõ".
4. **Bắt đầu thu ảnh webcam** theo `ban_giao_nhom/HUONG_DAN_THU_ANH_WEBCAM.md` (10–20 người, 3 phiên khác ngày, có người lạ). Ảnh webcam không dùng để train, nhưng cần cho bước 8 và mất thời gian thu.
5. ~~Chạy full train B1/B2~~: **xong 06/11** (`huan_luyen/Train_full_B1_B2.ipynb`). Lượt train song song cần gửi notebook, checkpoint và kết quả trên query che để đối chiếu đầy đủ.
6. ~~Bước 7~~: **xong 11/11** (`danh_gia_test/`). Kết luận: dùng **B0** cho demo webcam.
7. **Bước 8, phần còn lại:**
   - Chạy thử `demo/webcam_demo.py` và `demo/thu_anh_webcam.py` trước camera thật (chưa ai chạy; code mới kiểm trên ảnh/video).
   - Thu ảnh webcam 3 phiên khác ngày bằng `demo/thu_anh_webcam.py`, điền `nguoi_tham_gia.csv`.
   - Run All `webcam/Hieu_chinh_va_test_webcam.ipynb`: chọn ngưỡng bằng phiên 2, test bằng phiên 3. App demo tự đọc ngưỡng mới.
   - Viết báo cáo/slide, trình bày cả kết quả âm của B1/B2 trên ảnh thật; quay video demo dự phòng.

## Cấu trúc thư mục

```text
README.md                         ← file này
ban_giao_nhom/
  README_GOI_BAN_GIAO.md          cài môi trường và chạy lại
  huong_dan_trien_khai_hoan_chinh.pdf / .md   kế hoạch + phân công A/B/C
  build_guide_pdf.py              dựng lại PDF từ file .md
  requirements-eda-lock.txt       phiên bản thư viện đã dùng
  code/                           script baseline B0 (so 1 ảnh với nhiều ảnh)
  eda/EDA_kham_pha_du_lieu.ipynb  notebook EDA
  eda/notebook_outputs/           bảng CSV + hình PNG/SVG dùng cho slide
  qa/QA_nhan_trung_bounding_box.ipynb  notebook QA
  qa/notebook_outputs/            bảng, hình, contact sheet để duyệt
  qa/*_review_*.csv               log review (nhóm cập nhật khi duyệt)
  qa/group_decisions.csv          input quyết định theo ảnh, không bị ghi đè
  protocol/Chia_tap_va_protocol_danh_gia.ipynb  notebook split/protocol draft
  protocol/notebook_outputs/      vai trò ảnh, kiểm rò rỉ, feasibility, hash và JSON draft
  quyet_dinh_nhom/Cac_quyet_dinh_can_xac_nhan.ipynb  5 chính sách P1–P5 + trang duyệt gallery
  quyet_dinh_nhom/input/          nơi nhóm ghi quyết định (notebook chỉ đọc, không ghi đè)
  preprocessing/Preprocessing_thi_diem_validation.ipynb  pilot resize thẳng vs chạy lại detector
  preprocessing/notebook_outputs/ bảng, hình, cấu hình ứng viên, pixel pilot (NPZ)
  danh_gia_b0/B0_evaluator_validation.ipynb  evaluator 1:N + kết quả B0 trên validation
  danh_gia_b0/notebook_outputs/   DIR/FPIR, khoảng tin cậy, dự đoán từng query, ngưỡng validation, hình
  augmentation/Augmentation_che_co_kiem_soat.ipynb  manifest và quy tắc sinh che
  huan_luyen/README_PILOT.md      kết quả, checkpoint và thứ tự chạy lại
  huan_luyen/Pilot_train_B1_B2.ipynb  train pilot và đánh giá validation
  huan_luyen/Phan_tich_va_kiem_chung_pilot.ipynb  phân tích theo mask và kiểm resume
  huan_luyen/checkpoints/         encoder pilot (có trong gói bước 6 của Codex, không có trong gói này)
  huan_luyen/notebook_outputs/, analysis_outputs/  bảng, hình và báo cáo kiểm tra của pilot
  huan_luyen/Train_full_B1_B2.ipynb  train full B1/B2 trên 1.600 người
  huan_luyen/README_TRAIN_FULL.md kết quả train full, đối chiếu lượt song song, cách dùng model
  huan_luyen/checkpoints_full/    B1_best_encoder.pt, B2_best_encoder.pt (model đã lưu)
  huan_luyen/full_outputs/        bảng, hình, tóm tắt lượt train full
  huan_luyen/tai_model.py         nạp model đã lưu + trích embedding (cho bước 7, demo)
  danh_gia_test/Danh_gia_doc_lap_B0_B1_B2.ipynb  bước 7: DigiFace test, ROF, MFR2
  danh_gia_test/khoa_cau_hinh_buoc7.json  cấu hình khóa trước khi chấm test
  danh_gia_test/README_DANH_GIA_TEST.md  kết quả bước 7 và cách đọc
  danh_gia_test/notebook_outputs/ bảng, hình, tóm tắt lượt chạy
  demo/nhan_dien.py               pipeline dùng chung: tìm mặt, embedding, gallery, cửa sổ hiển thị
  demo/webcam_demo.py             app demo webcam (tên / Unknown / không thấy mặt, đổi B0–B3 bằng phím 1–4)
  cai_tien_B0_B3/                 chuỗi cải tiến B0 → B3: notebook 01–03, cai_tien.py, model B1 LFW, file khóa, README_CAI_TIEN.md
  demo/thu_anh_webcam.py          thu ảnh webcam đúng cấu trúc ba phiên
  demo/README_DEMO.md             cách thu ảnh, chạy demo, giới hạn cần nói khi demo
  do_toc_do/Do_toc_do_pipeline.ipynb  bước 8A: thời gian từng đoạn trên GPU/CPU
  webcam/Hieu_chinh_va_test_webcam.ipynb  bước 8B: ngưỡng (phiên 2) và test (phiên 3), chờ ảnh
  HUONG_DAN_THU_ANH_WEBCAM.md     kế hoạch thu ảnh webcam cho bước 8
data_acquisition/                 log và manifest khi tải DigiFace
data/sources/                     annotation ROF, metadata MFR2
data/raw/, data/benchmark_views/  ảnh gốc (~6 GB, KHÔNG có trong ZIP)
```

## Dữ liệu ảnh gốc

ZIP không chứa ảnh gốc (~5,3 GB `data/raw` + ~0,6 GB `data/benchmark_views`). Muốn chạy lại notebook, cần chép hai thư mục này vào đúng vị trí trên. Link chia sẻ: _(nhóm điền)_. Ảnh ROF/MFR2 là ảnh người nổi tiếng lấy từ web, chỉ dùng trong phạm vi môn học và không đăng công khai.
