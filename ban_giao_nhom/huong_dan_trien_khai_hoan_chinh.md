# NHẬN DIỆN KHUÔN MẶT KHI BỊ CHE KHUẤT
## Hướng dẫn triển khai và phân công nhóm

**Bàn giao 21/10/2026 · Cập nhật sau QA 27/10/2026 · Thị giác máy tính nâng cao**

Xây dựng ứng dụng webcam nhận diện người đã đăng ký khi đeo khẩu trang hoặc bị che một phần khuôn mặt. Hệ thống có quyền trả về **Unknown** khi chưa đủ bằng chứng và báo riêng khi không tìm được mặt.

### Câu hỏi cần chứng minh
- Fine-tune có giúp embedding tổng quát hóa sang người không xuất hiện trong fine-tune không?
- Thêm che nhân tạo có cải thiện so với fine-tune trên cùng dữ liệu nền không thêm che nhân tạo trong cùng ngân sách không?
- Kết quả trên che nhân tạo chuyển sang khẩu trang/kính thật đến mức nào; suy giảm ở bước tìm mặt hay nhận diện?
- Tại mức nhận nhầm người lạ đã chốt, hệ thống nhận đúng được bao nhiêu người đã đăng ký?

### Phạm vi thống nhất
Nhánh chính: MTCNN và InceptionResnetV1 pretrained VGGFace2 qua facenet-pytorch. Đầu vào xử lý một mặt; gallery ảnh rõ đã kiểm tra tay; query rõ hoặc che; có người lạ. So sánh B0 pretrained, B1 fine-tune dữ liệu nền không thêm che nhân tạo, B2 cùng dữ liệu nền có thêm che nhân tạo. ArcFace, triplet loss, nhiều frame, nhiều mặt và chống giả mạo là phần mở rộng sau khi nhánh chính hoàn chỉnh.

### Đọc tài liệu và sử dụng số liệu
Tài liệu này thay thế bản hướng dẫn 9 trang để triển khai; giữ nguyên bản gốc làm tham chiếu. Các con số ghi “đề xuất” là cấu hình thăm dò, không phải kết quả thực nghiệm hay cam kết chất lượng. Số ảnh thực tải, chất lượng dữ liệu và ảnh minh họa nằm trong notebook EDA và notebook QA đi kèm. Chỉ coi nguồn dữ liệu sẵn sàng khi log chứng minh tải, giải nén và đọc được.

### Trạng thái hiện tại
Đã xong: thu thập dữ liệu, EDA sâu (`eda/EDA_kham_pha_du_lieu.ipynb`) và QA nhãn/trùng/box (`qa/QA_nhan_trung_bounding_box.ipynb`). QA mới dừng ở mức **đề xuất**: 804 ảnh tạm giữ, 82 bản trùng phụ, 2.946 ảnh cần xem, 83.148 ảnh chưa phát hiện lỗi; chưa áp dụng vào dữ liệu. Crop ROF theo annotation là dữ liệu nguồn để kiểm tra, chưa là cache đầu vào model. Bước tiếp theo: nhóm duyệt bảng quyết định QA, rồi mới khóa split, preprocessing và augmentation.

### Điều kiện phù hợp môn học
Đề tài phù hợp nếu có thí nghiệm đối chứng, protocol người mới, open set, robustness và phân tích lỗi. Mức mới nghiên cứu còn hạn chế; nếu rubric đòi kiến trúc/phương pháp mới, cần trao đổi với giảng viên. Không bắt buộc B2 thắng B0; kết luận âm vẫn có giá trị khi đối chứng đúng và tái lập được.

<!-- PAGEBREAK -->
# 01 · Nguồn dữ liệu và phạm vi bằng chứng

### Vai trò của từng nguồn
| Nguồn | Quy mô tham khảo của tác giả | Vai trò dự kiến |
|---|---|---|
| CASIA-WebFace | Khoảng 0,5 triệu ảnh, 10 nghìn danh tính | Chưa tải được do quota nguồn; nhánh khôi phục tùy chọn về sau |
| DigiFace-1M | Tổng hợp; P1 2.000 danh tính × 72 ảnh; chọn EDA 40 ảnh/người | Nguồn fine-tune chính hiện tại; đánh giá domain gap thật/tổng hợp |
| ROF | Công bố 180 danh tính; 3.195 ảnh thường, 1.686 kính râm, 678 khẩu trang | Ứng viên đánh giá che thật chính; annotation hiện có 183 nhãn, 182 nhãn còn crop sau khi tải; cần QA |
| MFR2 | 269 ảnh, 53 danh tính; protocol tác giả 848 cặp | Benchmark phụ theo protocol cặp; không dùng train mạng lớn |
| Webcam nhóm | Đề xuất 10-20 người tự nguyện, 3 phiên | Kiểm tra gallery/query và demo thực tế |

**Nhánh hiện tại: DigiFace P1 tổng hợp; chốt eligibility sau EDA.** DigiFace có sẵn mũ, kính, che mắt và góc khó; condition chưa được gán nhãn, không gọi toàn bộ là “ảnh rõ”. CASIA gặp quota Google Drive. Với ROF crawl từ URL, ảnh gốc/annotation có thể khác gói benchmark tác giả; nhãn bbox cần QA. Số liệu thực nằm trong notebook EDA/QA, không lấy số công bố làm số đã tải.

**Đợt thu thực tế:** manifest EDA có 86.980 ảnh: DigiFace 80.000, MFR2 269, ROF 6.711 crop từ 6.296 ảnh gốc tải hợp lệ, thuộc 182 nhãn. Ảnh gốc ROF được giữ riêng, không cộng vào 86.980. Có một cặp crop trùng pixel mang hai nhãn ROF khác nhau, và ba ứng viên trùng tên giữa MFR2/ROF (Joe Biden, Justin Trudeau, Emmanuel/Immanuel Macron): cần xử lý trước khi khóa split. Đây là dữ liệu ứng viên chưa làm sạch, không phải tập test đã được xác minh.

### Quy trình thu thập có thể kiểm tra lại
1. Lập `sources.csv`: URL chính thức, ngày truy cập, cách truy xuất, phiên bản/commit, mục đích và điều kiện sử dụng.
2. Tải từ bản phát hành hoặc đường dẫn do tác giả công bố; lưu URL cuối, dung lượng, HTTP status và checksum nếu có. Không vượt đăng nhập hoặc tự cho rằng mọi mirror được phép phân phối lại.
3. Kiểm tra archive, giải nén vào vùng raw bất biến; đối chiếu số file và sample decoder. Ghi log lỗi, tải lại có giới hạn, tránh ghi đè bản gốc.
4. Tạo manifest từng ảnh và hash; chỉ báo “đã tải” khi file thực tồn tại, “đọc được” khi decoder thành công.
5. Lưu bộ chỉ mục tái tải trong gói nhóm. Ảnh gốc/trọng số lớn lưu riêng theo điều kiện nguồn và dung lượng, không nhét vào ZIP tài liệu.

### Giới hạn và sử dụng dữ liệu
Giấy phép code, ảnh và checkpoint là ba phần riêng. Dùng dữ liệu đúng phạm vi môn học và điều kiện nguồn; không suy quyền chia sẻ ảnh từ giấy phép thư viện. Webcam cần đồng thuận về mục đích, người được truy cập và thời điểm xóa. Không suy đoán tuổi, giới tính, sắc tộc từ khuôn mặt. Danh tính không xuất hiện trong fine-tune chưa chắc chưa xuất hiện trong pretraining VGGFace2; ghi rõ nếu không xác minh được.

<!-- PAGEBREAK -->
# 02 · EDA sâu: nguồn gốc và toàn vẹn

EDA phải tạo ra **bằng chứng cho quyết định**, không chỉ vài histogram. Giữ raw nguyên trạng; mọi thống kê có mẫu số, phạm vi và quy tắc tính. Phân biệt kết quả trên toàn bộ dữ liệu và trên mẫu chẩn đoán.

### Tầng 1 - Kiểm kê và đọc ảnh
- Đếm file đã tải, file ảnh ứng viên, file decode thành công/thất bại; bytes và định dạng. Báo số danh tính trước/sau đọc ảnh, ảnh không có nhãn và nhãn trống.
- Kiểm tra kích thước bằng decoder; ảnh 0 pixel, truncated, hỏng màu, RGBA/grayscale/CMYK, EXIF orientation và ảnh nén bất thường. Không âm thầm xoay hay chuyển màu trong raw.
- Kiểm tra cấu trúc nguồn: nhãn là thư mục hay annotation; ánh xạ điều kiện có thực sự được tài liệu nguồn xác nhận. Tên file chỉ là quy tắc parser, chưa là ground truth nếu chưa kiểm chứng.
- Lưu `decode_status`, exception tóm tắt, raw dimensions, file format, mode và orientation. Ảnh hỏng là dòng có lý do, không biến mất khỏi bảng số lượng.

### Tầng 2 - Trùng lặp, gần trùng, nhãn
- SHA-256 cho mọi ảnh để phát hiện trùng byte. Nếu cần, thêm hash pixel đã decode để phát hiện cùng ảnh lưu khác metadata; ghi rõ hai phép đo khác nhau.
- Perceptual hash để tìm ứng viên gần trùng; kiểm tra bằng mắt contact sheet. Crop, thay nền, ánh sáng, ảnh liên tiếp có thể gần trùng nhưng không phải file trùng.
- Báo số nhóm và số ảnh, cả trong cùng nhãn và giữa nhãn. Không tự gộp người chỉ vì pHash/embedding gần nhau; embedding không phải bằng chứng định danh độc lập.
- Lưu cặp ứng viên, khoảng cách hash, quyết định người kiểm, lý do và ngày kiểm. Nhãn mâu thuẫn phải giải quyết hoặc cách ly trước train.
- Nhóm cùng ảnh/phiên/chuỗi gần trùng phải đi cùng split. Nếu trùng giữa hai danh tính, xử lý nhãn trước khi chia theo identity.

### Sản phẩm bắt buộc
`sources.csv`, `download_log`, `manifest_raw.csv`, `decode_errors.csv`, `exact_duplicates.csv`, `near_duplicate_candidates.csv`, `duplicate_review.csv`, bảng số lượng đi qua từng bước. Manifest raw giữ cả các dòng bị loại; tạo cột trạng thái và manifest usable riêng.

### Diễn giải cho slide
Nêu số đo → hệ quả → quyết định dự kiến. Ví dụ: “Nhiều ảnh có alpha” dẫn tới kiểm tra chuyển RGB và nền, không dẫn trực tiếp tới kết luận model A tốt hơn model B. Con số thật và ví dụ minh họa phải lấy từ artifact EDA, không dùng số dự kiến trong tài liệu này.

<!-- PAGEBREAK -->
# 03 · EDA sâu: phân bố và điều kiện ảnh

### Tầng 3 - Độ phủ danh tính và điều kiện
| Phân tích | Cách báo và điều phải kiểm tra |
|---|---|
| Ảnh trên mỗi identity | Min, p10, median, p90, max; histogram và đường tích lũy; số identity chỉ có 1, 2, dưới 5 ảnh |
| Điều kiện che | Counts ảnh/identity của clean, mask, sunglasses, unknown; ma trận identity × condition |
| Mẫu paired | Số identity có clean và mask, clean và glasses, đủ cả ba; không dùng tổng 180 ROF cho mọi nhóm |
| Kích thước và tỷ lệ | Width/height, aspect ratio, số pixel; phân tầng theo nguồn và điều kiện |
| Dấu hiệu shortcut | Nền, watermark, biên ảnh, kiểu crop, nguồn, độ phân giải có gắn với nhãn/điều kiện không |
| Khác biệt nguồn | Phân bố độ sáng, kích thước, crop và hình ảnh; nhận diện domain gap tổng hợp/thật |

### Tầng 4 - Chất lượng ảnh bằng proxy
Đo brightness/luma median và quantiles, độ tương phản, tỷ lệ pixel gần tối/sáng, độ mờ proxy như variance of Laplacian. Đo trên ảnh gốc và crop chẩn đoán nếu có; ghi công thức, resize và màu trước tính. So sánh blur giữa ảnh kích thước khác nhau phải có độ phân giải tham chiếu hoặc nêu nhiễu do scale.

Proxy thấp/cao không tự động thành ảnh “xấu”. Ảnh che có texture khác ảnh rõ; detector score thấp không cho biết % che. Chưa đặt ngưỡng lọc blur/brightness chỉ từ histogram. Cần contact sheet biên phân phối và kiểm tra tác động số identity bị mất nếu lọc.

### Tầng 5 - Detector chẩn đoán có kiểm soát
- Cố định MTCNN config/version; ưu tiên đo toàn bộ bộ đánh giá nhỏ. Với nguồn train lớn, lấy mẫu phân tầng tái lập; công bố cách lấy và số mẫu từng nhóm.
- Báo zero/one/multiple faces, exception, confidence, bbox, diện tích mặt trên ảnh và face pixels; breakdown theo nguồn/điều kiện/kích thước.
- Confidence không phải xác suất tìm đúng; xác nhận bbox bằng mắt. Ảnh nhiều mặt không mặc định chọn mặt lớn nhất vì có thể nhầm identity mục tiêu.
- Kiểm tra có cần landmark/similarity alignment không bằng mẫu và ablation sau này. MTCNN crop/resize mặc định không đồng nghĩa alignment 5 điểm.
- Failure vẫn ở mẫu số toàn hệ thống. Không loại ảnh khó khỏi benchmark để làm đẹp tỷ lệ nhận diện.

### Hình cho slide
Ít nhất: biểu đồ counts theo nguồn/điều kiện; histogram ảnh mỗi người; ma trận độ phủ điều kiện; độ phân giải và face size; blur/brightness theo nhóm; bảng detector status; contact sheet raw và failure. Hình có tiêu đề, đơn vị, N và chú thích phạm vi mẫu.

<!-- PAGEBREAK -->
# 04 · EDA sâu: kiểm tra tay và quyết định

### Kiểm tra tay có thiết kế
Tạo mẫu ngẫu nhiên cố định và mẫu trường hợp biên riêng: ảnh mờ/tối/sáng, mặt nhỏ, detector thất bại, nhiều mặt, điều kiện che, nhóm gần trùng và identity ít ảnh. Đề xuất 100-200 ảnh cho pilot nếu đủ dữ liệu; đây là ngân sách QA, không phải cỡ mẫu bảo đảm thống kê. Tăng mẫu khi còn phát hiện lỗi mới; lưu selection seed và image_id.

Mỗi ảnh cần các nhãn kiểm: đúng người theo nguồn có thể xác minh hay chưa, đúng condition, mặt mục tiêu, bbox/crop hợp lệ, che tự nhiên hay nhân tạo, ảnh có thể dùng và lý do. Không khẳng định label sạch nếu chỉ kiểm tra vài ảnh. Nhãn danh tính nghi vấn cần hai người xem khi có thể; lưu bất đồng, không ép quyết định bằng cosine.

### Bảng nối EDA với pipeline
| Bằng chứng thực đo | Quyết định cần duyệt | Cách xác nhận tiếp |
|---|---|---|
| Nhiều nhóm gần trùng/ảnh liên tiếp | Gom nhóm trước split, giới hạn ảnh mỗi nhóm | Kiểm tra giao nhau giữa split bằng id/hash |
| Identity thiếu ảnh/thiếu điều kiện | Chọn quy tắc eligibility và paired subset | Báo ảnh/identity bị loại và coverage còn lại |
| Mặt nhỏ, crop lệch hoặc zero face cao | Rà detector/resize/alignment | Ablation validation; giữ end-to-end failure |
| Blur/darkness lệch giữa nguồn | Thử augmentation nhẹ có lý do | Kiểm tra tensor, sample QA; không tự xóa test |
| Nguồn tổng hợp khác ảnh thật | Tách source, dùng real eval độc lập | Báo domain gap và giới hạn kết luận |
| Ít unknown/ít mask | Giảm kỳ vọng ước lượng, bổ sung thu thập | Báo interval rộng; không nêu FAR cực thấp |

### Tránh kết luận vượt dữ liệu
EDA giải thích vì sao cần chuẩn hóa đầu vào, kiểm soát split và thử robust training. EDA **không chứng minh** InceptionResnetV1 tối ưu. Lựa chọn nhánh này ban đầu dựa trên checkpoint PyTorch, tính khả thi fine-tune và baseline tái lập; chất lượng phải do thí nghiệm xác nhận. Không suy fairness nhân khẩu học khi không có nhãn tin cậy.

### Gate kết thúc EDA
A bàn giao báo cáo, bảng quyết định keep/review/exclude, ví dụ lỗi và manifest; B kiểm tra khả năng đọc/nhãn/tensor dự kiến; C kiểm tra mẫu số và feasibility gallery/query/unknown. Chốt nguồn train và protocol dự kiến bằng văn bản. **Chưa sinh cache crop đầu vào model/augmentation hàng loạt hoặc train B1/B2 trước khi cả nhóm duyệt bước tiếp theo.**

<!-- PAGEBREAK -->
# 05 · Chia tập và protocol gallery/query

### Tách người trước augmentation
Đề xuất pilot: chọn tối đa 2.000 identity đủ ảnh; giới hạn 20-40 ảnh/người bằng seed cố định; ví dụ 1.600 train, 200 validation, 200 test. Các số này chỉ là điểm bắt đầu, phải điều chỉnh theo EDA, độ phủ điều kiện và tài nguyên. Không áp số này lên ROF hoặc bộ nhỏ.

- Split theo canonical identity trước crop/che. Ảnh gốc và mọi biến thể cùng split; nhóm trùng/gần trùng không vượt split.
- Train-monitor được lấy từ ảnh không gần trùng của training identities; chỉ để theo dõi overfit, không phải khả năng nhận diện người mới.
- Validation/test nội bộ có identity tách hoàn toàn khỏi fine-tune và nhau. Trong mỗi tập, chia known và unknown theo identity; unknown không có bất cứ ảnh nào trong gallery.
- Known có gallery và query khác original_id/nhóm gần trùng, ưu tiên khác session. Webcam: phiên đăng ký, hiệu chỉnh, test riêng; người unknown validation/test cũng riêng.
- Kiểm tra identity chồng lặp giữa nguồn khi có metadata; không tự cho rằng khác tên dataset là khác người. Lưu các overlap chưa xác minh được.

### Thẻ protocol phải chốt trước chạy
Ghi source/version, split hash, seed, eligibility, số known, số unknown, số ảnh gallery/người, số query mỗi điều kiện, số session/identity, quy tắc chọn query và failure. Ghi rõ N ảnh và N người độc lập; không chỉ tổng frames.

Đề xuất nội bộ: gallery 3-5 ảnh rõ/người nếu đủ ảnh; webcam 5-10 ảnh rõ/người. Không ép benchmark thiếu ảnh đáp ứng số này. Query che được so với **gallery clean**. Gallery có thêm masked là thí nghiệm riêng.

### ROF/MFR2: tách hai loại báo cáo
ROF là benchmark che thật chính; chỉ subset có đủ điều kiện mới được so paired. Nếu giữ toàn bộ làm test cuối thì không dùng kết quả ROF để chọn model/epoch/threshold/augmentation. Ngưỡng nội bộ được hiệu chỉnh trên nguồn validation tách biệt; báo domain shift nếu chuyển ngưỡng sang ROF. Nếu cần ROF-validation, chia identity trước, khóa ROF-test và công bố đây là protocol tùy chỉnh.

MFR2 bổ sung verification theo pairs/folds tác giả. Nếu có cross-validation, tuân thủ ngưỡng của từng fold và báo riêng; không dùng ngưỡng 1:1 đó để gán tên ở webcam 1:N. Không trộn số liệu protocol tác giả với protocol nội bộ thành một con số.

<!-- PAGEBREAK -->
# 06 · Tiền xử lý sau khi duyệt EDA

### Hợp đồng đầu vào nhánh chính
Đọc ảnh, xử lý EXIF orientation theo quy tắc cố định, chuyển RGB; MTCNN `image_size=160` và standardization đúng facenet-pytorch; encoder trả embedding 512 chiều, L2 normalized. Không thêm ImageNet normalization hoặc standardize hai lần. DigiFace trong mẫu kiểm có ảnh gốc 112 × 112: tăng lên 160 là nội suy, không khôi phục chi tiết bị thiếu; giữ giới hạn này trong phân tích.

MTCNN crop/resize mặc định không đồng nghĩa similarity alignment 5 điểm. Nếu thêm alignment, phải định nghĩa transform, kiểm tra landmarks và áp nhất quán cho B0/B1/B2. Detector cố định trong thí nghiệm recognition, tránh thay hai yếu tố cùng lúc.

**Phân biệt crop nguồn với toàn cảnh.** Nhiều ảnh DigiFace/MFR2/ROF đã crop sát; tìm mặt lại có thể cho bbox vượt khung, chưa có nghĩa detector sai. Không mặc định redetect mọi crop. Thí điểm so direct-resize 160 với redetect-crop trên audit/validation; xác minh alignment theo metadata, rồi khóa cùng chính sách cho B0/B1/B2. Webcam toàn cảnh vẫn cần detector và log failure riêng.

### Chính sách ảnh đầu vào
| Tình huống | Xử lý và log |
|---|---|
| Không tìm thấy mặt | `no_face`; ghi vào mẫu số end-to-end, không tạo embedding giả |
| Nhiều mặt | `multiple_faces`; review ở train; demo một mặt yêu cầu đầu vào đúng phạm vi |
| Bbox/crop không hợp lệ | `invalid_crop`; giữ ảnh và lý do để phân tích |
| Decode hỏng | `decode_error`; lỗi dữ liệu đầu vào ghi riêng |
| Face nhỏ hoặc score thấp | Flag theo config đã duyệt; chỉ lọc train khi có lý do, không âm thầm loại test |

### Cache và tính nhất quán
Giữ raw bất biến. Cache crop dạng pixel RGB thông thường; standardization khi load. Không ghi tensor có giá trị âm thành JPEG. Lưu preprocess version, detector config và hash raw cho mỗi crop. Khi thay preprocessing/model, xây lại gallery và hiệu chỉnh lại ngưỡng.

Đối chiếu online và offline trên cùng ảnh: shape, dtype, range, crop coordinates và sai khác tensor/embedding; ghi tolerance theo định dạng cache. Nếu dùng JPEG, sai khác nén phải được đo; PNG thuận lợi hơn cho kiểm tra tương đương.

### Hai nhánh đánh giá cần tách
**Recognition có vị trí cố định:** crop chuẩn → thêm che → encoder. Đo embedding khi biết vị trí mặt; mẫu số và eligibility đã cố định.

**End-to-end:** ảnh che thật hoặc ảnh gốc thêm che → detector → crop → encoder → gallery. Tính mọi đầu vào và detector failures. Không suy kết quả trên crop chuẩn thành chất lượng webcam.

### Nghiệm thu
Cấu hình preprocessing được version hóa; không có normalize kép; tensor online/offline đã kiểm tra; ảnh minh họa crop đủ góc/điều kiện; bảng counts giữ/loại và lý do; split và gallery/query hashes không đổi.

<!-- PAGEBREAK -->
# 07 · Tạo che khuất có kiểm soát

### Thiết kế biến đổi
MaskTheFace: chạy thử trên ảnh gốc trước crop khi cần landmarks/ngữ cảnh. Nếu nguồn chỉ có face crop, kiểm tra tỷ lệ đặt đúng trên crop; không giả định tương đương ảnh gốc. Kiểm tra tay 100-200 ảnh đủ điều kiện/góc nếu có trước khi sinh lớn; lưu cả ảnh sinh thất bại.

Che mắt, che bên trái/phải hoặc vùng dưới: tạo binary mask riêng, lưu vị trí, diện tích, seed, version và original_id. “0/10/20/30/40%” là cấu hình thử; 0% nghĩa là không thêm che, không bảo đảm dữ liệu nền vốn không che. Mức cao là stress test, không hứa nhận diện được.

Tỷ lệ che cần mẫu số cụ thể: vùng mặt quy ước có segmentation hoặc toàn crop. Nếu dùng toàn crop thì gọi đúng là **% diện tích crop bị che**. Không dùng detector confidence để suy phần trăm mặt bị che. Với che thật không có mask ground truth, dùng condition của nguồn, không bịa mức phần trăm.

### Đối chứng B0/B1/B2
| Mốc | Dữ liệu và cập nhật | Vai trò |
|---|---|---|
| B0 | Pretrained nguyên trạng, `eval()`, không cập nhật | Chất lượng ban đầu |
| B1 | Fine-tune dữ liệu nền, augmentation nhẹ, không thêm che nhân tạo | Tác động học thêm dữ liệu nền |
| B2 | Cùng khởi tạo B1; cùng dữ liệu nền + che nhân tạo | Tác động của che bổ sung |
| B3 tùy chọn | B2 kết hợp nhiều frame | Lợi ích thông tin thời gian, báo riêng |

Đề xuất B2 ban đầu có xác suất 50% giữ ảnh nền và 50% thêm che nhân tạo; một lần lấy mẫu chỉ sinh một phiên bản. Đây không phải tỷ lệ clean/occluded thật vì dữ liệu nền có thể đã che. Không nhân đôi dataset khiến B2 có gấp đôi optimizer steps. Có thể thử 25/50/75% trên validation với cùng ngân sách. Chỉ gọi B1 “thuần ảnh rõ” nếu dùng subset clean đã QA riêng.

### Cố định đánh giá
Validation/test nhân tạo được sinh một lần với seed và mask lưu lại, mọi model dùng cùng ảnh. Không sinh ngẫu nhiên mỗi lần đo. Nối clean và occluded theo original_id để so paired; báo số ảnh hợp lệ của từng điều kiện. Lưu kết quả cả trên tập eligible cố định và toàn pipeline để lộ survivor bias nếu detector chỉ giữ ảnh dễ.

Synthetic kiểm tra tác động có kiểm soát; ROF/MFR2 kiểm tra chuyển sang che thật. Không kết luận augmentation có lợi chỉ vì FaceNet khác ArcFace: đối chứng chính là B2 so B1 cùng kiến trúc và ngân sách.

<!-- PAGEBREAK -->
# 08 · Mô hình và fine-tune

### Warm-up kỹ thuật
Chạy 20-50 training identities, vài batch; kiểm tra label mapping, tensor, loss, gradient, optimizer và checkpoint load lại. Có thể cố overfit tập nhỏ để debug nhưng không đưa làm kết quả nghiên cứu. B0 inference/evaluator có thể phát triển trên tập nhỏ trước khi train đầy đủ.

### Học embedding bằng classifier tạm
Gắn classifier có số lớp bằng identity trong TRAIN và dùng cross-entropy. Warm-up classifier mới 1-2 epoch khi encoder frozen; bước này chưa cải thiện encoder. Sau đó mở block cuối/projection, tăng phần mở chỉ khi validation có lý do.

Classifier train không biết nhãn identity mới. Khi đánh giá người mới, bỏ classifier và lấy embedding L2 normalized. Không dùng accuracy classifier trên identity validation mới để chọn model. Nếu dùng API có `classify=True`, xác nhận đường forward embedding đúng, không lấy logits làm embedding.

Phần frozen phải giữ BatchNorm ở eval để running statistics không âm thầm đổi khi gọi `train()`. Kiểm tra số tham số trainable, gradient và BN state; dropout train/eval theo nhánh thích hợp. Lưu rõ policy cho B1/B2.

### Cấu hình đề xuất để thăm dò
| Tham số | Điểm khởi đầu, chưa tối ưu |
|---|---|
| Optimizer | AdamW; weight decay 1e-4 |
| Learning rate | Classifier 1e-3; encoder 1e-5 hoặc 3e-5 |
| Batch | 16 hoặc 32, chốt sau đo VRAM |
| Epoch | Trần 5-15; early stopping patience 3 validation |
| Seeds | 1 pilot; 3 seed cấu hình cuối nếu tài nguyên cho phép |

B1/B2 cùng pretrained initialization, số optimizer steps, effective batch size, lịch validation và ngân sách tuning. Không khởi tạo B2 từ checkpoint B1 nếu không có đối chứng train tương ứng. Nếu giảm tài nguyên, giảm cả hai nhánh; không để B2 được thử nhiều cấu hình hơn mà bỏ qua.

### Chọn checkpoint và báo cáo
Chọn theo embedding validation người mới: DIR rank-1 tại mục tiêu FPIR đã chốt; công bố tiêu chí phụ và tie-break trước chạy. Nếu mục tiêu không ước lượng được, dừng để sửa protocol, không chọn bằng test. Giữ kết quả giảm chất lượng; báo trung bình/độ lệch chuẩn qua seed nếu có.

Lưu layer mở/frozen, BN policy, trainable count, optimizer/scheduler, steps, seed, GPU, môi trường, manifest/preprocessing hashes, checkpoint SHA-256 và log. ArcFace head/triplet loss chỉ mở rộng sau; thay loss cần đối chứng B1/B2 cùng loss.

<!-- PAGEBREAK -->
# 09 · Bài toán chính và chọn ngưỡng

### Nhận diện 1:N có người lạ
Với query q, tính cosine với mọi template gallery; lấy score lớn nhất và tên tương ứng. Chấp nhận khi score ≥ threshold, ngược lại Unknown. Gallery template: trung bình các embedding ảnh đã L2 normalize rồi L2 normalize lại. Mọi model dùng cùng ảnh và quy tắc.

**DIR rank-1 = số query known được gán đúng tên và vượt ngưỡng / tổng query known.** Wrong identity, rejection và detection failure của known đều không là thành công. **FPIR = số query unknown bị chấp nhận thành bất kỳ tên nào / tổng query unknown.** Dùng mẫu số toàn đầu vào cho end-to-end, và báo riêng trên query có crop hợp lệ.

Detector failure ở unknown không gây false accept nhưng vẫn phải báo tỷ lệ bao phủ detector; FPIR thấp nhờ detector không hoạt động không phải hệ thống tốt. Báo đồng thời DIR, FPIR và detector failure theo condition.

### Mục tiêu calibration đề xuất
Đề xuất mục tiêu FPIR validation 5% cho đồ án; chốt sau khi biết số unknown độc lập. Không đủ mẫu thì nêu không ước lượng ổn định, bổ sung dữ liệu hoặc công bố giới hạn. Ví dụ 100 unknown query cho bước tỷ lệ 1%, chưa bảo đảm interval hẹp; frames cùng clip không tính độc lập.

Quét threshold trên validation; chọn threshold cho DIR cao nhất trong các threshold có FPIR thực đo không vượt mục tiêu, tie-break ưu tiên ngưỡng cao hơn. Báo actual FPIR và interval: tiêu chí theo point estimate không phải bảo đảm FPIR thật ≤ 5%. Nếu dùng upper confidence bound để chọn, phải ghi thành quy tắc khác và chốt trước chạy.

Mỗi model có threshold riêng theo cùng quy tắc và gallery size. Threshold 1:1 không tự chuyển sang 1:N vì maximum qua gallery làm đổi phân phối. Đổi gallery cần calibration tương ứng. Validation tổng hợp chỉ hiệu chỉnh trên miền tổng hợp; webcam cần calibration người thật tách phiên. Không coi mục tiêu FPIR synthetic là bảo đảm cho ảnh thật.

### Khóa trước test
Khóa checkpoint, preprocessing, gallery/query list, seed, threshold và metric trong config có hash. Không dùng test để chọn epoch, threshold, augmentation hoặc mẫu đẹp. Nếu xem lỗi test rồi sửa model, gọi lần chạy đó là phát triển; cần test độc lập mới nếu muốn đánh giá xác nhận.

### Verification là báo cáo bổ sung
1:1 đo ROC-AUC, TAR tại FAR định trước; ghi số positive/negative pairs và số lỗi. Không ước lượng FAR cực thấp từ vài trăm cặp. MFR2 pairs/folds theo tác giả báo riêng với protocol nội bộ; không có lỗi trên mẫu nhỏ không đồng nghĩa rủi ro bằng không.

<!-- PAGEBREAK -->
# 10 · Đánh giá, khoảng tin cậy và tốc độ

### Bảng kết quả tối thiểu
Mỗi dòng ghi model, seed, source, condition, gallery size, số known/unknown queries và identities, detector failures, correct accepted, wrong identity, rejected known, accepted unknown, DIR, FPIR và latency. Lưu prediction từng query để truy ngược lỗi; không chỉ ảnh chụp bảng tổng.

### So sánh công bằng theo điều kiện
- Clean đã kiểm tra tay, dữ liệu nền chưa nhãn che, synthetic mắt/dưới/trái/phải, mask thật, glasses thật và webcam báo riêng. Không đổi tên dữ liệu nền thành clean hoặc gộp vô điều kiện các nguồn khác nhau.
- So sánh paired B1/B2 trên cùng query; với ROF clean-mask/glasses chỉ dùng identity có đủ điều kiện cho phân tích paired, đồng thời báo coverage toàn nguồn.
- Conditional recognition chỉ trên crop hợp lệ là chỉ số bổ sung. Nếu từng model có tập crop khác nhau, giao/common subset phải được mô tả; end-to-end cùng đầu vào là căn cứ thực tế.
- Tránh survivor bias: thống kê ảnh bị loại theo condition và độ khó; detector fail không được biến mất khỏi bảng.

### Uncertainty và đơn vị độc lập
Báo 95% confidence interval cho tỷ lệ khi phù hợp, nêu phương pháp và N. Wilson/binomial chỉ phù hợp khi giả định độc lập hợp lý; nhiều query cùng người/clip có tương quan. Khi có nhiều query/person, ưu tiên bootstrap theo identity/session, cùng resample cho chênh lệch paired B2-B1. Với 3 seed báo thêm biến thiên giữa lần train; không trộn seed như thêm người độc lập.

Không chỉ nói “tăng 3%”: ghi điểm phần trăm, N, interval và chi phí clean performance. Nếu interval rộng hoặc chênh lệch nhỏ, kết luận bằng chứng còn yếu. Không có lỗi vẫn cần interval; benchmark nhỏ không chứng minh an toàn triển khai.

### Đo latency
Warm-up trước; cố định phần cứng, độ phân giải, số mặt và batch=1 khi mô phỏng webcam. Đo detect/crop, embedding, gallery search và end-to-end; report median/p95 và số lần đo. CUDA cần đồng bộ trước/sau đo nếu dùng timer CPU. Ghi rõ đã bao gồm I/O và vẽ UI hay chưa; display FPS khác inference FPS.

### Phân tích lỗi có ích cho thuyết trình
Phân loại detector không thấy, crop sai, mặt quá nhỏ, unknown bị nhận nhầm, known bị từ chối, wrong identity, che/ánh sáng/góc. Show vài trường hợp tiêu biểu và tỷ lệ, không chỉ chọn ảnh đẹp. Kết luận nối về EDA và giả thuyết ban đầu; phân biệt nguyên nhân đã kiểm chứng với suy đoán cần ablation.

<!-- PAGEBREAK -->
# 11 · Hợp đồng tích hợp và cấu trúc tệp

### Cấu trúc dự án đề xuất
`data/raw/<source>/` dữ liệu gốc; `data/manifests/` bảng ảnh và split; `data/processed/<preprocess_version>/` crop; `configs/` protocol/train/preprocess; `src/` data/model/evaluation/demo; `checkpoints/`; `reports/eda/`; `reports/experiments/`; `tests/smoke/`. Không commit dataset, thông tin người thu webcam hoặc trọng số lớn vào Git mặc định.

### Các bảng phải thống nhất trước khi làm riêng
| Tệp | Các trường cốt lõi |
|---|---|
| manifest.csv | image_id, source, source_identity, canonical_identity, original_id, path, sha256, decode_status, width, height, condition, session_id |
| qa_manifest.csv | image_id, face_count, detector_score, bbox, face_size, blur_proxy, brightness_proxy, crop_status, exclude_reason, review_status |
| variants.csv | variant_id, original_id, condition, transform_seed, transform_version, mask_path, mask_area, area_denominator, preprocessing_version |
| splits.json | version, seed, manifest_hash, train_ids, val_ids, test_ids, duplicate_group_policy |
| gallery/query.csv | image_id, canonical_identity, partition, role, condition, session_id, gallery_id, protocol_version |
| predictions.csv | run_id, query_id, status, predicted_identity, top_score, threshold, is_known, correct, detector_status, latency_ms |

Trường chưa biết dùng null/unknown theo schema; không trộn “không có” với giá trị 0. Quy định path tương đối project root, UTF-8, bbox dạng `[x1,y1,x2,y2]` theo pixel ảnh sau orientation; field metadata và version được lưu rõ. `role`: gallery/known_query/unknown_query; `partition`: val/test.

### API chung đề xuất
`prepare_face(image, config) -> status, crop_tensor, detection_meta`.
`encode(crop_batch) -> float32[N,512]`, RGB chuẩn hóa đúng một lần, `eval()` và L2 normalized.
`build_gallery(embeddings, identities) -> templates, identity_order`.
`predict(embedding, templates, threshold) -> identity_or_unknown, cosine`.

Không trả vector zero khi detector lỗi; truyền status để evaluator tính đúng mẫu số. Một adapter chung load B0/B1/B2; checkpoint chứa encoder state và metadata, không bắt evaluator dùng classifier head. Ghi device, dtype, model/preprocess hash vào run metadata. Chỉ thay giao diện sau khi A/B/C cùng duyệt schema.

<!-- PAGEBREAK -->
# 12 · Phân công A/B/C và kiểm tra chéo

| Phụ trách | Công việc chủ động, làm song song | Bàn giao và người kiểm |
|---|---|---|
| A - Dữ liệu | Thu thập, provenance, EDA sâu; tạo mẫu nhỏ đúng schema sớm; sau duyệt mới split/preprocess/che | Data card, manifests, EDA, decision log, QA; C kiểm split/test, B kiểm preprocessing |
| B - Mô hình | B0 trên mẫu nhỏ; adapter encoder; warm-up; B1/B2 đối chứng; log và model card | Script/config/checkpoint/log; A+C kiểm dữ liệu và kết quả |
| C - Đánh giá/demo | Viết protocol/evaluator bằng B0 ngay; calibration; báo cáo latency; webcam một mặt | Gallery/query/pairs, thresholds, predictions, metrics/demo; A rà protocol, B rà embedding |
| Cả nhóm | Chốt schema, đọc kết quả, tích hợp, tái lập, slide/clip dự phòng | README, environment, giới hạn, nguồn; một người giữ đầu mối tích hợp |

Tên thật nhóm tự điền bên cạnh A/B/C. Đề xuất C điều phối tích hợp vì giữ evaluator/demo; nhóm có thể đổi trước T0. Mỗi task có owner chính, reviewer, deadline thực và trạng thái trên bảng chung. Không chờ A hoàn thiện toàn bộ dataset mới cho B/C bắt đầu code.

### Phiếu tài nguyên cần điền sau đo
GPU local đã kiểm tra: NVIDIA GeForce RTX 4060 Laptop GPU, VRAM 8188 MiB theo nvidia-smi (xấp xỉ 8 GB). CPU/RAM/disk và giờ sử dụng/ngày: nhóm điền; máy giữ dữ liệu: nhóm điền; quản lý training: B. Thời gian một epoch chưa đo, không suy từ tên GPU.

B đo pilot: load time, batch khả thi, peak VRAM, seconds/step, một epoch mất bao lâu, dung lượng checkpoint/cache. Dự tính giờ B1/B2 từ steps thực đo, cộng validation và các seed; lưu dự tính và thực tế. Không cam kết thời gian train trước khi đo GPU.

### Ngân sách và ngưỡng dừng
Chốt tối đa identity/ảnh/epoch/steps, số cấu hình và số seed theo phần cứng. B1/B2 cùng ngân sách; nếu OOM thì giảm batch cả hai hoặc dùng gradient accumulation cùng effective batch. Nếu hết thời gian, ưu tiên hoàn chỉnh B0/B1/B2 một seed, protocol thật và phân tích lỗi trước ArcFace/UI đồ thị.

### Tiêu chí từng nhiệm vụ
Một việc chỉ “xong” khi có artifact, lệnh chạy lại, dữ liệu/config hash và reviewer xác nhận. Tải được archive chưa là EDA xong; loss giảm chưa là recognition tốt; demo hiện tên chưa là đánh giá open set đạt.

<!-- PAGEBREAK -->
# 13 · Mốc triển khai và quyết định chuyển bước

T0 là thời điểm **nguồn dữ liệu đọc được và B0 inference chạy thật trên mẫu đã xác nhận**, không phải ngày tài liệu được viết. Lịch dưới là thời lượng đề xuất, nhóm điền ngày sau khi biết deadline môn và tài nguyên; có thể chồng lấp công việc B/C trên mẫu nhỏ.

| Mốc tương đối | Owner | Phụ thuộc và điều kiện nghiệm thu |
|---|---|---|
| Trước/T0 | A+B | Tải/đọc ảnh, provenance; B0 có cosine và status thực; môi trường lưu lại |
| T0 + 1-3 ngày | A; B/C song song | EDA sâu + manual QA + decision log; schema và evaluator thử trên B0 |
| Gate EDA | A+B+C | Duyệt nguồn, eligibility, protocol và preprocessing; nhóm quyết định bước tiếp |
| +1-2 ngày sau duyệt | A+B | Split không leakage; crop QA; tensor online/offline; synthetic cố định |
| +1 ngày | B+C | B0 đầy đủ; protocol/metrics/calibration hoạt động; pilot resource thực đo |
| Theo pilot đo | B | B1/B2 cùng ngân sách, log/checkpoint load lại; chọn bằng validation |
| +1 ngày | C+A | Khóa toàn bộ config rồi chạy test; báo uncertainty, errors, latency |
| +1-2 ngày | C+cả nhóm | Webcam, README, máy khác chạy lại, slide và video dự phòng |

### Gate và phương án khi vướng
- Nguồn train không truy cập/điều kiện không phù hợp: ghi fail minh bạch; thử nguồn dự phòng, EDA lại và đổi tên thí nghiệm. Không giả làm đã có CASIA.
- Không đủ ảnh cho gallery/query/unknown: hạ quy mô, bổ sung nguồn/thu webcam; công bố protocol nhỏ và độ bất định. Không dùng cùng ảnh hai vai trò.
- Detector fail nhiều ở che thật: giữ failure trong kết quả; sửa ở validation hoặc báo giới hạn. Không “cứu” test bằng lọc ảnh sau xem lỗi.
- Fine-tune giảm chất lượng: kiểm tensor/BN/label/learning rate trên validation; giữ log; B0 vẫn là demo hợp lệ nếu kết luận đúng.
- Thời gian thiếu: bỏ B3/ArcFace/đồ thị UI trước; giữ đối chứng, open set, EDA và khả năng chạy lại.

### Quy trình ngắn gọn để nhóm chọn bước
**Thu thập → kiểm toàn vẹn + EDA sâu → duyệt quyết định → khóa split/protocol → preprocessing + QA → B0 đầy đủ → sinh che có kiểm soát → B1/B2 → calibration/chọn bằng validation → khóa → test + phân tích → webcam + báo cáo/slide.**

Đã xong thu thập, EDA và QA đề xuất. Bước tiếp theo là nhóm duyệt bảng quyết định QA, rồi khóa split/protocol và làm preprocessing thí điểm. Chưa xử lý hay train hàng loạt trước khi cả nhóm thống nhất.

<!-- PAGEBREAK -->
# 14 · Chạy script baseline đi kèm

Script: `code/face_occlusion_baseline_compare.py`. Mục tiêu là so một reference với nhiều queries bằng B0, xuất cosine và trạng thái detector. Không chọn threshold, không train, không gán tên. Reference lỗi hoặc nhiều mặt phải được từ chối; query lỗi được ghi vào output, không bỏ âm thầm.

### Chuẩn bị và chạy trong PowerShell
Tạo môi trường Python 3.10/3.11 riêng; cài theo requirements của script và chạy `pip check`. Ví dụ bên dưới dùng facenet-pytorch 2.6.0 theo nhánh gốc; phiên bản được xác nhận trong README môi trường của gói. Không nâng torch/torchvision tùy tiện; CUDA cần tương thích driver.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r code/requirements-baseline.txt
.venv\Scripts\python -m pip check
.venv\Scripts\python code/face_occlusion_baseline_compare.py `
  --reference reference.jpg `
  --queries same_person.jpg masked_same_person.jpg other_person.jpg `
  --output b0_smoke.json --device cpu
.venv\Scripts\python -m pip freeze > environment.txt
```

Lần đầu model có thể tải trọng số. Chuẩn bị ảnh đúng một mặt, có quyền sử dụng. Ba query ví dụ chỉ là smoke test, không đủ chọn threshold hay kết luận accuracy.

**Đã kiểm chứng inference thật:** CUDA RTX 4060, torch 2.2.2+cu121, facenet-pytorch 2.6.0; 4/4 query hợp lệ được tính cosine, self-cosine gần 1 trong sai số 1e-5. Một query cố tình không tồn tại được ghi lỗi, nên exit code 1 của ca này là dự kiến. Bằng chứng ở `code/validation/b0_smoke_real.json` và `validation_summary.json`. Đây là kiểm tra chức năng; chưa đo latency benchmark hoặc chất lượng nhận diện.

### Kiểm tra output và tiếp tục
Ảnh cùng người/khác người có cosine chỉ là ví dụ chẩn đoán. Nếu cùng người score thấp, xem ảnh, crop, màu và normalize trước; không chỉnh ngưỡng từ ba ảnh. So tensor/crop và detector status; lưu môi trường Python/OS/GPU. Script này là baseline reference-query, chưa là evaluator 1:N hoặc training framework.

### Tối thiểu để gửi nhóm
PDF này, nguồn Markdown có thể sửa, script + requirements/README, notebook EDA và QA kèm hình/bảng, provenance/manifest và quy trình chạy lại. Dataset/trọng số lớn có đường dẫn riêng và điều kiện truy cập. Thành viên điền tên A/B/C, deadline thực, tài nguyên và reviewer trước triển khai toàn bộ.

<!-- PAGEBREAK -->
# 15 · Demo, báo cáo và tài liệu tham khảo

### Demo cần có
Thu webcam tự nguyện, đề xuất 10-20 người nếu khả thi, ba phiên: đăng ký, calibration, test. Unknown riêng, không đưa vào gallery. Test clean, mask thật, glasses, tay che, quay nhẹ và đổi ánh sáng. Chỉ thu đủ phục vụ môn học; quản lý quyền truy cập và xóa như đã thống nhất.

UI tối thiểu: tên/Unknown/không tìm thấy mặt, ảnh tham chiếu và cosine; cosine không phải xác suất đúng. Có lựa chọn B0/model fine-tune và reset. Đồ thị score theo thời gian là mở rộng. Thử người đeo khẩu trang ngay từ khi bước vào, người lạ và ánh sáng xấu.

Nếu smoothing, xóa nhãn cũ khi mất dấu/hết hạn; không giữ tên từ frame rõ rồi gọi là nhận diện che thành công. Báo per-frame và multi-frame riêng. Chưa có liveness, demo không dùng như hệ thống mở cửa an toàn.

### Mạch slide đề xuất
Vấn đề và giả thuyết → dữ liệu/provenance → EDA và lỗi thật → quyết định split/preprocess/model có bằng chứng → protocol B0/B1/B2 → kết quả kèm N/CI → phân tích thất bại và domain gap → demo → giới hạn/hướng tiếp. Mỗi hình EDA cần một thông điệp dẫn tới lựa chọn kỹ thuật, không chỉ mô tả biểu đồ.

### Nguồn kỹ thuật chính
- facenet-pytorch, README và checkpoint: https://github.com/timesler/facenet-pytorch
- Inference notebook: https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
- Fine-tune notebook: https://github.com/timesler/facenet-pytorch/blob/master/examples/finetune.ipynb
- Encoder source: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py
- Video/tracking: https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb
- ROF: https://github.com/ekremerakin/RealWorldOccludedFaces
- MaskTheFace/MFR2: https://github.com/aqeelanwar/MaskTheFace
- CASIA trong danh mục InsightFace: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
- DigiFace-1M: https://github.com/microsoft/DigiFace1M
- ArcFace PyTorch: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

Notebook classifier là ví dụ kỹ thuật; phải thay split và evaluator theo tài liệu để đo người mới. Khi sử dụng trong báo cáo, ghi phiên bản/commit và bài báo gốc của tác giả. Link không phải bằng chứng file đã tải hoặc điều kiện sử dụng đã được kiểm tra. Mọi kết quả phải truy về dữ liệu, seed, checkpoint, preprocessing, threshold và gallery.
