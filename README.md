# Phân tích khuôn mặt thời gian thực

App webcam nhận diện **cảm xúc, nhóm tuổi, giới tính ước lượng** và **theo dõi ID** từng khuôn mặt. Chạy cục bộ trên CPU, không gửi ảnh ra ngoài. Đồ án của nhóm Vinh, Tiến, Toàn.

## Chạy app

**Cần:** Windows, [Python 3.12 hoặc 3.13](https://www.python.org/downloads/) 64-bit (có `py`), webcam, Internet lần đầu, khoảng 3 GB trống.

Mở PowerShell tại thư mục dự án:

```powershell
.\setup.bat      # chỉ chạy lần đầu, cài vào venv, mất khoảng 10–15 phút
.\run_app.bat    # mở app
```

Mở `http://localhost:8501` nếu trình duyệt không tự mở, rồi vào tab **Live** và bấm **Bắt đầu**.

| Gặp lỗi | Cách xử lý |
| --- | --- |
| `KHONG tao duoc moi truong` | Cài Python 3.12 hoặc 3.13. Python 3.14 trở lên chưa dùng được với TensorFlow |
| `Không mở được webcam` | Đóng ứng dụng khác đang dùng camera, kiểm tra quyền camera của Windows |
| Muốn cài lại từ đầu | Xóa thư mục `venv`, rồi chạy lại `.\setup.bat` |

Kiểm tra code (67 test, khoảng 30 giây):

```powershell
.\venv\Scripts\python.exe scripts/check_sync.py
.\venv\Scripts\python.exe -m unittest discover -s tests
```

## Tình trạng hiện tại

| Hạng mục | Kết quả |
| --- | --- |
| Cài mới bằng `setup.bat`, 67 test, app khởi động | Đạt, đã thử trên bản sao sạch |
| Tốc độ, 1 mặt, 640×480 | **30 FPS** (trước đó 23). Chưa tính camera và Streamlit |
| Tốc độ, 3 mặt | **13 FPS** (trước đó 9) |
| Theo dõi ID | Không tráo ID trên 5 kịch bản mô phỏng (trước đó 13 lần) |
| Xác suất giới tính | Đã hiệu chuẩn, ECE từ 0,094 xuống 0,009 |
| Giới tính, UTKFace | Đúng 85,4% |
| Tuổi, UTKFace | Sai trung bình 10,3 năm, đúng nhóm tuổi 47,4% |
| Cảm xúc, RAF-DB validation | Accuracy 84,7%, macro-F1 0,819 |
| **Cảm xúc, webcam thật** | **Chưa đo trên bản mới.** Bản cũ, 1 người, quay chính diện: chỉ 34,4% |

**Cần biết trước khi demo:**

- Chưa thử app với người thật trước webcam. Camera mới được kiểm tra đọc khung ở 30 FPS.
- Fear, sad, disgust từng đúng 0 khung trên webcam ở bản cũ, và model mặc định vẫn dự đoán như trước.
- Giới tính rất kém với trẻ em và thiếu niên nam: bé trai 0–12 tuổi đúng **48,6%**, nam 13–19 đúng **52,9%**, nữ 50+ đúng **74,3%**. Hệ thống chưa thể gọi là công bằng.
- Tuổi và giới tính chỉ là ước lượng từ hình dáng khuôn mặt, không xác định bản dạng giới của một người. Không dùng để ra quyết định về ai.
- Bản mới dùng CPU nhiều gấp đôi (khoảng 440%), máy sẽ nóng hơn.

## Cách hệ thống hoạt động

```text
Webcam → luôn lấy khung mới nhất
       → MediaPipe tìm mặt
       → Crop vuông, xoay cho thẳng mắt
       → Gán ID, giữ ID 10 khung khi mặt tạm mất
       → Chấm điểm chất lượng mặt
       → Cảm xúc: EfficientNet → hiệu chỉnh bằng blendshape → làm mượt
       → Tuổi, giới tính: mỗi 5 khung → hiệu chuẩn → gộp 12 lần gần nhất
       → Vẽ lên màn hình
```

Thiếu model tuổi/giới tính thì cảm xúc vẫn chạy. Thiếu hoặc sai file ONNX thì app tự dùng Keras, chậm hơn nhưng cùng kết quả.

## Đọc code ở đâu

Bắt đầu từ `app.py`, rồi `core/pipeline.py`.

```text
app.py              giao diện Streamlit
core/               phần lõi app dùng
models/             model và tham số hiệu chuẩn
scripts/            train, đánh giá, đo tốc độ; app không gọi
tests/              test tự động
results/            số liệu đo được
```

| File trong `core/` | Việc |
| --- | --- |
| `pipeline.py` | Ghép mọi bước, đọc camera |
| `detectors.py` | Tìm mặt, landmark, crop |
| `tracker.py` | Giữ ID qua các khung |
| `quality.py` | Điểm chất lượng, làm mượt |
| `predict.py` | Dự đoán cảm xúc |
| `fusion.py` | Hiệu chỉnh cảm xúc bằng blendshape |
| `attributes.py` | Tuổi, giới tính, hiệu chuẩn |
| `ui_helpers.py` | Nhóm tuổi, vẽ khung, dòng thống kê |
| `preprocessing.py`, `selection.py`, `webcam_data.py` | Tiền xử lý, luật chọn checkpoint, đọc ghi dữ liệu |

| Nhóm script | File |
| --- | --- |
| Thử trên webcam | `condition_test.py`, `run_conditions.py`, `summarize_conditions.py`, `summarize_by_expr.py` |
| Đánh giá cảm xúc | `evaluate_ferplus.py`, `experiment_emotion_head.py`, `evaluate_rafdb.py`, `evaluate_webcam.py` |
| Đánh giá tuổi, giới tính | `evaluate_utkface_attributes.py`, `calibrate_age_groups.py`, `calibrate_attributes.py`, `select_attribute_config.py` |
| Đo pipeline | `evaluate_tracking.py`, `evaluate_quality.py`, `benchmark_pipeline.py` |
| ONNX | `export_onnx.py`, `check_onnx.py` |
| Train RAF-DB | `prepare_rafdb.py`, `prepare_webcam.py`, `train_rafdb.py`, `train_fusion.py` |
| Khác | `prepare_public_data.py`, `summarize_final.py`, `check_sync.py` |

## Việc còn lại của nhóm

**1. Thử app với người thật:** một mặt và hai mặt, đổi chỗ, che mặt, đổi biểu cảm, thiếu sáng, quay ngang, chạy 30 phút, rút camera khi đang chạy.

**2. Chọn model mặc định.** Sidebar của app có 3 lựa chọn:

| Trong sidebar | Tên trong lệnh | Macro-F1 FER+ | Fear / Sad / Disgust |
| --- | --- | --- | --- |
| EfficientNet (baseline), mặc định | `efficientnet` | 0,461 | 0,25 / 0,45 / 0,08 |
| hsemotion (thử nghiệm) | `hsemotion` | 0,496 | 0,24 / 0,56 / 0,25 |
| EfficientNet — ferplus | `efficientnet_ferplus` | 0,623 | 0,44 / 0,58 / 0,29 |

FER+ là ảnh xám 48×48, không đại diện webcam, nên phải so trực tiếp. Chạy lệnh dưới cho đủ 7 biểu cảm (`surprise fear disgust happy sad angry neutral`), nên từ 3 người trở lên. Lệnh chỉ ghi số liệu, không lưu ảnh:

```powershell
.\venv\Scripts\python.exe scripts/condition_test.py --who nguoi01 --expr sad --model efficientnet --fusion models/efficientnet_fusion.json --out results/webcam_baseline.csv
.\venv\Scripts\python.exe scripts/condition_test.py --who nguoi01 --expr sad --model hsemotion --out results/webcam_hsemotion.csv
.\venv\Scripts\python.exe scripts/condition_test.py --who nguoi01 --expr sad --model efficientnet_ferplus --out results/webcam_ferplus.csv
```

**3. RAF-DB test:** chưa chạy lại được vì thiếu dataset. Mọi đánh giá khác không phụ thuộc vào file này.

## Chi tiết kết quả

Máy đo: Intel i5-10210U, 16 GB RAM, chỉ CPU. Chọn cấu hình chỉ trên validation, test chỉ để báo cáo, seed 42. Số gốc nằm trong `results/`.

| Chỉ số | Trước | Sau |
| --- | --- | --- |
| Latency 1 mặt, đầy đủ | 43 ms | 33 ms |
| Latency 3 mặt, đầy đủ | 108 ms | 79 ms |
| Bộ nhớ | 531 MB | 442 MB |
| Trễ tuổi/giới tính (mô phỏng) | 84–95 khung | 28–31 khung |
| Dao động tuổi (mô phỏng) | 1,39 năm | 1,87 năm (kém hơn) |

"Độ trễ 6 giây" cũ là độ dài cửa sổ gộp tuổi/giới tính (12 × 15 khung), không phải độ trễ xử lý.

UTKFace test, 18.780 ảnh, CI 95% bootstrap theo ảnh: MAE tuổi 10,27 (10,14–10,41), nhóm tuổi 47,4% (46,7–48,1), macro-F1 43,9% (43,2–44,6), giới tính 85,4% (84,8–85,9). Nhãn không phải tự khai và không có ID người. Không đánh giá theo chủng tộc vì không có dữ liệu tự khai hợp lệ.

FER+ test, 7.000 ảnh, CI 95% của macro-F1: EfficientNet 0,448–0,474; HSEmotion 0,481–0,512; `efficientnet_ferplus` 0,598–0,650. HSEmotion không làm mặc định vì trên validation CI của mức chênh chứa 0. `efficientnet_ferplus` không làm mặc định vì lớp cuối học trên chính kiểu ảnh FER+.

Đã thử nhưng không có lợi: TTA lật ngang, focal loss, label smoothing, weighted CE, oversample, MixUp, fit lại ngưỡng nhóm tuổi, fine-tune EfficientNet trên RAF-DB. Điểm chất lượng nhận ra ảnh hỏng rõ, nhưng trên ảnh tự nhiên điểm thấp không đi kèm sai nhiều hơn.

## Chạy lại đánh giá

Dataset không kèm trong dự án. Lệnh đầu tải khoảng 1 GB về `data/`.

```powershell
.\venv\Scripts\python.exe scripts/prepare_public_data.py --utkface --ferplus --near-duplicates
.\venv\Scripts\python.exe scripts/evaluate_utkface_attributes.py --workers 4 --crop-scale 1.5 --output-dir results/utkface_rerun
.\venv\Scripts\python.exe scripts/calibrate_attributes.py results/utkface_rerun/predictions.csv --output results/genderage_calibration_rerun.json
.\venv\Scripts\python.exe scripts/select_attribute_config.py
.\venv\Scripts\python.exe scripts/evaluate_ferplus.py --limit 3000 --output results/ferplus_rerun
.\venv\Scripts\python.exe scripts/experiment_emotion_head.py
.\venv\Scripts\python.exe scripts/evaluate_tracking.py
.\venv\Scripts\python.exe scripts/evaluate_quality.py
.\venv\Scripts\python.exe scripts/check_onnx.py
.\venv\Scripts\python.exe scripts/benchmark_pipeline.py
.\venv\Scripts\python.exe scripts/summarize_final.py
```

Chạy `summarize_final.py` sau cùng vì nó cần đặc trưng do `experiment_emotion_head.py` tạo.

Train lại trên RAF-DB cần dataset riêng: `prepare_rafdb.py` → `train_rafdb.py --manifest ...` → `train_fusion.py --manifest ...` → `evaluate_rafdb.py`. Xuất lại ONNX theo hướng dẫn trong `scripts/export_onnx.py`, rồi chạy `check_onnx.py`.

## Nguồn và giấy phép

Chỉ dùng cho nghiên cứu và học tập. Không phân phối dataset hoặc ảnh mặt kèm dự án.

| Thành phần | Nguồn | Lưu ý |
| --- | --- | --- |
| MediaPipe Face Landmarker | Google | |
| `genderage.onnx` | InsightFace `buffalo_l` | Trọng số chỉ cho nghiên cứu phi thương mại |
| UTKFace | Bản sao `nu-delta/utkface` | Chỉ nghiên cứu phi thương mại |
| FER+ | Bản sao `deanngkl/ferplus-7cls` | Chưa đối chiếu ảnh và giấy phép với nguồn gốc |
| HSEmotion | Dự án `face-emotion-recognition`, Apache-2.0 | Huấn luyện trên AffectNet, kiểm tra điều khoản trước khi dùng thương mại |
| `efficientnet.keras` | Do nhóm bàn giao | Chưa rõ dữ liệu train ban đầu |
