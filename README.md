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

Kiểm tra code (73 test, khoảng 30 giây):

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
| Cảm xúc, RAF-DB test (3.068 ảnh, crop như app) | Accuracy 80,3%, macro-F1 0,722 (EfficientNet + fusion, lề 0) |
| Cảm xúc, RAF-DB validation | Accuracy 90,6%, macro-F1 0,882 (EfficientNet + fusion; lạc quan vì model đã thấy ảnh train) |
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
       → Chấm điểm chất lượng mặt; mặt kém thì dừng ở đây và báo lý do
       → Cảm xúc: EfficientNet → hiệu chỉnh bằng blendshape → làm mượt
       → Tuổi, giới tính: mỗi 5 khung → hiệu chuẩn → gộp 12 lần gần nhất
       → Vẽ lên màn hình
```

**Kiểm tra chất lượng đầu vào** (bật mặc định trong app, tắt ở thanh bên). Mặt bị chặn thì không gọi model, không vào bộ làm mượt và không ghi vào thống kê.

| Trạng thái | Khi nào | Xử lý |
| --- | --- | --- |
| `LARGE_POSE` | max(\|yaw\|, \|pitch\|, \|roll\|) > 30° | Không dự đoán, yêu cầu nhìn về camera |
| `TOO_FAR` | Mặt rộng dưới 80 px | Không dự đoán, yêu cầu đến gần |
| `LOW_LIGHT` | Độ sáng trung bình vùng mặt dưới 45 | Không dự đoán, yêu cầu tăng sáng |
| `BLURRY` | Độ nét (Crete) dưới 0,25 | Không dự đoán, yêu cầu giữ yên |
| `NEED_ALIGNMENT` | Góc 15–30° | Bắt buộc căn chỉnh rồi dự đoán |
| `LOW_CONFIDENCE` | Xác suất cao nhất dưới 0,45, hoặc mặt 80–119 px | Dự đoán nhưng ghi "Không chắc chắn" |

Yaw, pitch lấy từ ma trận xoay của MediaPipe, còn roll lấy từ đường nối hai mắt. Ngưỡng độ nét và độ sáng chỉ được kiểm trên RAF-DB train (chặn nhầm 0% và 0,7% ảnh thường), **chưa kiểm trên webcam**. Căn chỉnh 2D chỉ sửa được roll, không sửa được yaw hay pitch. `Pipeline(quality_gate=False)` là mặc định cho script đánh giá, để kết quả cũ không đổi.

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
| Đánh giá cảm xúc | `evaluate_rafdb.py`, `evaluate_webcam.py` |
| Đánh giá tuổi, giới tính | `evaluate_utkface_attributes.py`, `calibrate_age_groups.py`, `calibrate_attributes.py`, `select_attribute_config.py` |
| Đo pipeline | `evaluate_tracking.py`, `evaluate_quality.py`, `benchmark_pipeline.py` |
| ONNX | `export_onnx.py`, `check_onnx.py` |
| Train RAF-DB | `prepare_rafdb.py`, `prepare_webcam.py`, `train_rafdb.py`, `train_fusion.py` |
| Khác | `prepare_public_data.py`, `summarize_final.py`, `check_sync.py` |

## Việc còn lại của nhóm

**1. Thử app với người thật:** một mặt và hai mặt, đổi chỗ, che mặt, đổi biểu cảm, thiếu sáng, quay ngang, chạy 30 phút, rút camera khi đang chạy.

**2. So model trên webcam.** Sidebar của app có 2 lựa chọn; số dưới là RAF-DB test, 3.068 ảnh (`results/rafdb_test/`):

| Trong sidebar | Tên trong lệnh | Accuracy | Macro-F1 | Fear / Sad / Disgust (F1) |
| --- | --- | --- | --- | --- |
| EfficientNet (baseline), mặc định | `efficientnet` | 79,8% | 0,713 | 0,58 / 0,77 / 0,48 |
| hsemotion (thử nghiệm) | `hsemotion` | 70,6% | 0,563 | 0,33 / 0,78 / 0,36 |

RAF-DB không phải webcam, nên vẫn cần so trực tiếp. Chạy lệnh dưới cho đủ 7 biểu cảm (`surprise fear disgust happy sad angry neutral`), nên từ 3 người trở lên. Lệnh chỉ ghi số liệu, không lưu ảnh:

```powershell
.\venv\Scripts\python.exe scripts/condition_test.py --who nguoi01 --expr sad --model efficientnet --fusion models/efficientnet_fusion.json --out results/webcam_baseline.csv
.\venv\Scripts\python.exe scripts/condition_test.py --who nguoi01 --expr sad --model hsemotion --out results/webcam_hsemotion.csv
```

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

RAF-DB test, 3.068 ảnh, CI 95% bootstrap theo ảnh của macro-F1: EfficientNet 0,713 (0,689–0,736), HSEmotion 0,563 (0,540–0,584); mức chênh 0,122–0,177, nằm hẳn trên 0 nên giữ EfficientNet làm mặc định. HSEmotion nhận ảnh RAF-DB 100×100 phóng lên 224×224, khác ảnh AffectNet nó được train. Dữ liệu train ban đầu của `efficientnet.keras` chưa rõ, nhưng trên validation tách từ RAF-DB train nó đạt 89,3% so với 79,8% trên test (`results/rafdb_validation/`), tức gần như chắc đã thấy ảnh train. Vì vậy chỉ số test mới đáng tin.

**Lề crop 0, không phải 0,15.** Ảnh RAF-DB lúc train được cắt sát mặt. Cũng 2.740 ảnh test, cắt như app với lề 0 đạt 80,6%, lề 0,15 chỉ còn 72,9%; HSEmotion cũng tốt nhất ở lề 0 (70,0% so với 67,9%). Vì vậy app và fusion dùng lề 0; fusion đã train lại trên RAF-DB cắt lề 0 (`prepare_rafdb.py --margin 0`). Kết quả: `results/rafdb_test_fusion/`.

Đã thử nhưng không có lợi: TTA lật ngang, fit lại ngưỡng nhóm tuổi, fine-tune EfficientNet trên RAF-DB. Điểm chất lượng nhận ra ảnh hỏng rõ, nhưng trên ảnh tự nhiên điểm thấp không đi kèm sai nhiều hơn.

## Chạy lại đánh giá

Dataset không kèm trong dự án. Cảm xúc dùng RAF-DB: giải nén `RAF-DB.zip` vào `data/rafdb/` sao cho có `data/rafdb/DATASET/train/1..7` và `data/rafdb/DATASET/test/1..7`. Lệnh `prepare_public_data.py` tải UTKFace về `data/`.

```powershell
.\venv\Scripts\python.exe scripts/evaluate_rafdb.py --data-dir data/rafdb/DATASET --models efficientnet hsemotion --split test --output results/rafdb_test_rerun
.\venv\Scripts\python.exe scripts/check_onnx.py
.\venv\Scripts\python.exe scripts/prepare_public_data.py --utkface
.\venv\Scripts\python.exe scripts/evaluate_utkface_attributes.py --workers 4 --crop-scale 1.5 --output-dir results/utkface_rerun
.\venv\Scripts\python.exe scripts/calibrate_attributes.py results/utkface_rerun/predictions.csv --output results/genderage_calibration_rerun.json
.\venv\Scripts\python.exe scripts/select_attribute_config.py
.\venv\Scripts\python.exe scripts/evaluate_tracking.py
.\venv\Scripts\python.exe scripts/evaluate_quality.py
.\venv\Scripts\python.exe scripts/benchmark_pipeline.py
.\venv\Scripts\python.exe scripts/summarize_final.py
```

Train lại trên RAF-DB cần dataset riêng: `prepare_rafdb.py` → `train_rafdb.py --manifest ...` → `train_fusion.py --manifest ...` → `evaluate_rafdb.py`. Xuất lại ONNX theo hướng dẫn trong `scripts/export_onnx.py`, rồi chạy `check_onnx.py`.

## Nguồn và giấy phép

Chỉ dùng cho nghiên cứu và học tập. Không phân phối dataset hoặc ảnh mặt kèm dự án.

| Thành phần | Nguồn | Lưu ý |
| --- | --- | --- |
| MediaPipe Face Landmarker | Google | |
| `genderage.onnx` | InsightFace `buffalo_l` | Trọng số chỉ cho nghiên cứu phi thương mại |
| UTKFace | Bản sao `nu-delta/utkface` | Chỉ nghiên cứu phi thương mại |
| RAF-DB | Bản nhóm có sẵn (`RAF-DB.zip`, 12.271 ảnh train, 3.068 ảnh test) | Chỉ nghiên cứu phi thương mại theo điều khoản của tác giả RAF-DB |
| HSEmotion | Dự án `face-emotion-recognition`, Apache-2.0 | Huấn luyện trên AffectNet, kiểm tra điều khoản trước khi dùng thương mại |
| `efficientnet.keras` | Do nhóm bàn giao | Chưa rõ dữ liệu train ban đầu |
