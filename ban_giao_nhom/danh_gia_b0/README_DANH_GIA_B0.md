# Bước 5: Evaluator 1:N và baseline B0 trên validation

Mở `B0_evaluator_validation.ipynb` để xem đầy đủ code, hình và diễn giải. Notebook chạy từ kernel mới, 11/11 cell code, không lỗi; tính lại toàn bộ hai lần cho cùng kết quả.

## Thiết lập

- **Model:** B0 = InceptionResnetV1 pretrained VGGFace2, không fine-tune.
- **Dữ liệu:** DigiFace validation, 100 người quen (3 ảnh gallery + 37 query mỗi người) và 100 người lạ (40 query mỗi người), tổng 8.000 ảnh. Không dùng train/test, ROF, MFR2.
- **Chuẩn bị ảnh:** A = resize thẳng (mặc định theo P5), B = chạy lại MTCNN rồi crop.
- **Đánh giá** theo hợp đồng trong `protocol_draft.json`: template = trung bình embedding gallery; chọn ngưỡng sao cho FPIR thực đo ≤ 5% và DIR cao nhất (P4); mọi query đều nằm trong mẫu số, kể cả ảnh lỗi; khoảng tin cậy bootstrap theo người.

## Kết quả

| | A: resize thẳng | B: chạy lại MTCNN |
|---|---|---|
| Ảnh chuẩn bị được | 8.000/8.000 | 7.789/8.000 |
| Người đăng ký được | 100/100 | 97/100 |
| Rank-1 bỏ qua ngưỡng (tham khảo) | 82,9% | 79,7% |
| Ngưỡng cosine | 0,693 | 0,677 |
| **DIR tại FPIR 5%** | **57,2%** (CI 95%: 52,8–61,6%) | **54,7%** (CI 95%: 49,8–59,4%) |

- A hơn B 2,4 điểm phần trăm DIR; khoảng tin cậy của chênh lệch là 0,0–4,9 điểm, tức nghiêng về A nhưng bằng chứng còn yếu. Trên tập ảnh mà cả hai cách đều chuẩn bị được, A vẫn hơn (58,0% so với 55,9%), nên khác biệt không chỉ do detector bỏ sót ảnh. Kết quả này ủng hộ lựa chọn P5.
- Với B0, cứ khoảng 100 query người quen thì 57 được nhận đúng, 41 bị từ chối (điểm dưới ngưỡng) và khoảng 2 bị nhầm sang người khác.
- Lỗi điển hình: mặt nghiêng hẳn, quá tối, bị khăn/kính VR/tóc che. Đây là chỗ B1/B2 cần cải thiện.

## Tệp đầu ra

- `notebook_outputs/tables/04_operating_point_fpir5.csv`: ngưỡng, DIR, FPIR và đếm từng loại kết quả.
- `notebook_outputs/tables/05_bootstrap_ci_by_identity.csv`: khoảng tin cậy và chênh lệch A − B.
- `notebook_outputs/tables/08_predictions.csv`: dự đoán từng query (schema dùng chung cho B1/B2).
- `notebook_outputs/b0_validation_thresholds.json`: ngưỡng validation của B0, **chưa áp lên test**.
- `notebook_outputs/figures/`: đường DIR–FPIR, phân bố điểm, ví dụ lỗi (dùng được cho slide).
- `notebook_outputs/cache/`: embedding đã tính, không gửi kèm ZIP.

## Giới hạn

Ngưỡng được chọn trên chính tập validation nên DIR ở đây hơi lạc quan; con số không thiên lệch sẽ có ở bước 7 (test). DigiFace là ảnh render, nên kết quả này chưa nói gì về ảnh che thật (ROF/MFR2) hay webcam.
