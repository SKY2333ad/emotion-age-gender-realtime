# Phân tích khuôn mặt thời gian thực

App webcam nhận diện **5 cảm xúc** (ngạc nhiên, vui, buồn, tức giận, bình thản), **giới tính ước lượng** và **theo dõi ID** từng khuôn mặt. Chạy cục bộ trên CPU, không gửi ảnh ra ngoài. Đồ án của nhóm Vinh, Tiến, Toàn.

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

Kiểm tra code (62 test):

```powershell
.\venv\Scripts\python.exe scripts/check_sync.py
.\venv\Scripts\python.exe -m unittest discover -s tests
```

## Web AI Smart Photo Booth

Website chụp 3 ảnh theo chủ đề (Ngầu, Sinh nhật, Kinh dị, Giáng sinh), dùng cùng pipeline cảm xúc, giới tính và cổng chất lượng ở trên qua API `/api/analyze`. Trình duyệt chạy thêm MediaPipe FaceLandmarker để lấy blendshape, gộp với xác suất EfficientNet (0,65 / 0,35). Filter khuôn mặt vẽ bằng Canvas bám theo 478 điểm mốc, chạy sẵn không cần cài thêm: mỗi cảm xúc có một filter cho nam và một filter cho nữ (10 filter, `dist/ar/filters/expressionMatrix.js`); chủ đề chỉ đổi không khí nền (mũ tiệc, tuyết, dơi...). Giới tính chưa đủ chắc (dưới 0,75) thì chỉ hiện viền màu theo cảm xúc. Filter nâng cao DeepAR (chủ đề Ngầu) và thay nền phía sau là tùy chọn, cần License Key DeepAR: mở `http://localhost:4173/#setup` để nhập key một lần, key chỉ lưu trong `localStorage` của trình duyệt. Chưa có key thì giao diện không hiện hai mục này.

```powershell
.\run_photo_booth.bat           # mở http://127.0.0.1:4173, dùng chung venv với app Streamlit
.\STOP_AI_PHOTO_BOOTH.bat       # dừng server
```

`START_PUBLIC_PHOTO_BOOTH.bat` mở thêm đường hầm ngrok ra Internet nếu máy đã cài ngrok và có authtoken. Server chỉ nghe `127.0.0.1` và không ghi ảnh xuống đĩa. Mã nguồn web nằm trong `sites/ai-smart-photo-booth/`; test phía web chạy bằng `npm test` trong thư mục đó.

## Tình trạng hiện tại

| Hạng mục | Kết quả |
| --- | --- |
| Tốc độ, 1 mặt, 640×480, đủ chức năng | **22 FPS** (46 ms/khung, chạy Keras). Chưa tính camera và Streamlit |
| Tốc độ, 3 mặt, đủ chức năng | **9 FPS** (116 ms/khung) |
| Theo dõi ID | Không tráo ID trên 5 kịch bản mô phỏng |
| Xác suất giới tính | Đã hiệu chuẩn, ECE từ 0,094 xuống 0,009 |
| Giới tính, UTKFace | Đúng 85,4% |
| Cảm xúc, RAF-DB test (2.834 ảnh, crop như app) | Accuracy 84,9%, macro-F1 0,818 (EfficientNet 5 lớp, lề 0) |
| Cảm xúc, RAF-DB validation | Accuracy 92,9%, macro-F1 0,922 (dùng để chọn epoch nên lạc quan) |
| **Cảm xúc, webcam thật** | **Chưa đo** |

**Cần biết trước khi demo:**

- Chưa đo độ chính xác trên webcam với người thật.
- Trên RAF-DB, angry và sad là hai lớp yếu nhất (F1 0,75 và 0,82); trên webcam chưa có số.
- Giới tính rất kém với trẻ em và thiếu niên nam: bé trai 0–12 tuổi đúng **48,6%**, nam 13–19 đúng **52,9%**, nữ 50+ đúng **74,3%**. Hệ thống chưa thể gọi là công bằng.
- Giới tính chỉ là ước lượng từ hình dáng khuôn mặt, không xác định bản dạng giới của một người. Không dùng để ra quyết định về ai.
- App dùng khoảng 200% CPU và 530 MB RAM khi chạy.

## Cách hệ thống hoạt động

```text
Webcam → luôn lấy khung mới nhất
       → MediaPipe tìm mặt
       → Crop vuông, xoay cho thẳng mắt
       → Gán ID, giữ ID 10 khung khi mặt tạm mất
       → Chấm điểm chất lượng mặt; mặt kém thì dừng ở đây và báo lý do
       → Cảm xúc: EfficientNet 5 lớp → làm mượt
       → Giới tính: mỗi 5 khung → hiệu chuẩn → gộp 12 lần gần nhất
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

Yaw, pitch lấy từ ma trận xoay của MediaPipe, còn roll lấy từ đường nối hai mắt. Các ngưỡng là giá trị ban đầu, **chưa hiệu chỉnh trên webcam**. Căn chỉnh 2D chỉ sửa được roll, không sửa được yaw hay pitch. `Pipeline(quality_gate=False)` là mặc định cho script đánh giá, để kết quả cũ không đổi.

**Model cảm xúc.** `efficientnet.keras` là model 5 lớp (`surprise, happy, sad, angry, neutral`), fine-tune bằng `train_rafdb.py` trên RAF-DB train đã bỏ fear và disgust; epoch 3 được chọn theo validation (`results/training/rafdb5/`). App chạy model bằng Keras trên CPU.

`genderage.onnx` trả cả tuổi lẫn giới tính trong một lần chạy; dự án chỉ dùng giới tính. Thiếu model giới tính thì cảm xúc vẫn chạy.

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
| `attributes.py` | Giới tính, hiệu chuẩn |
| `ui_helpers.py` | Vẽ khung, dòng thống kê |
| `preprocessing.py`, `selection.py`, `webcam_data.py` | Tiền xử lý, luật chọn checkpoint, đọc ghi dữ liệu |

| Nhóm script | File |
| --- | --- |
| Thử trên webcam | `condition_test.py`, `run_conditions.py`, `summarize_conditions.py`, `summarize_by_expr.py` |
| Đánh giá cảm xúc | `evaluate_rafdb.py`, `evaluate_webcam.py` |
| Đánh giá giới tính | `evaluate_utkface_attributes.py`, `calibrate_attributes.py` |
| Đo pipeline | `evaluate_tracking.py`, `evaluate_quality.py`, `benchmark_pipeline.py` |
| Train RAF-DB | `prepare_rafdb.py`, `prepare_webcam.py`, `train_rafdb.py` |
| Khác | `prepare_public_data.py`, `summarize_final.py`, `check_sync.py` |

## Việc còn lại của nhóm

**1. Thử app với người thật:** một mặt và hai mặt, đổi chỗ, che mặt, đổi biểu cảm, thiếu sáng, quay ngang, chạy 30 phút, rút camera khi đang chạy.

**2. Đo model trên webcam.** Số dưới là RAF-DB test, 2.834 ảnh (`results/rafdb_test/`):

| Trong sidebar | Tên trong lệnh | Accuracy | Macro-F1 | Sad / Angry (F1) |
| --- | --- | --- | --- | --- |
| EfficientNet (baseline), mặc định | `efficientnet` | 84,8% | 0,819 | 0,82 / 0,75 |

RAF-DB không phải webcam, nên vẫn cần so trực tiếp. Chạy lệnh dưới cho đủ 5 biểu cảm (`surprise happy sad angry neutral`), nên từ 3 người trở lên. Lệnh chỉ ghi số liệu, không lưu ảnh:

```powershell
.\venv\Scripts\python.exe scripts/condition_test.py --who nguoi01 --expr sad --model efficientnet --out results/webcam_baseline.csv
```

## Chi tiết kết quả

Máy đo: Intel i5-10210U, 16 GB RAM, chỉ CPU. Chọn cấu hình chỉ trên validation, test chỉ để báo cáo, seed 42. Số gốc nằm trong `results/`.

| Chỉ số | Giá trị | Nguồn |
| --- | --- | --- |
| Latency 1 mặt, đầy đủ | 46 ms (p95 62 ms) | `results/benchmark_pipeline.json` |
| Latency 3 mặt, đầy đủ | 116 ms | `results/benchmark_pipeline.json` |
| Bộ nhớ | 530 MB | `results/benchmark_pipeline.json` |
| Trễ giới tính khi người đổi (mô phỏng) | 31 khung | `results/tracking_evaluation.json` |

UTKFace test, 18.780 ảnh, CI 95% bootstrap theo ảnh: giới tính đúng 85,4% (84,8–85,9) (`results/final_summary.json`). Nhãn không phải tự khai và không có ID người. Không đánh giá theo chủng tộc vì không có dữ liệu tự khai hợp lệ.

Model được fine-tune từ checkpoint nhóm bàn giao, vốn chưa rõ dữ liệu train ban đầu và gần như chắc đã thấy ảnh RAF-DB train. Validation đạt 92,9% so với 84,8% trên test (`results/rafdb_validation/`, `results/rafdb_test/`), nên chỉ số test mới đáng tin.

**Lề crop 0.** Ảnh RAF-DB lúc train được cắt sát mặt, nên app dùng lề 0 để ảnh lúc chạy giống ảnh lúc train. Kết quả khi cắt ảnh test giống app: `results/rafdb_test_app/`.

Điểm chất lượng nhận ra ảnh hỏng rõ, nhưng trên ảnh tự nhiên điểm thấp không đi kèm sai nhiều hơn.

## Chạy lại đánh giá

Dataset không kèm trong dự án. Cảm xúc dùng RAF-DB: giải nén `RAF-DB.zip` vào `data/rafdb/` sao cho có `data/rafdb/DATASET/train/<số lớp>` và `data/rafdb/DATASET/test/<số lớp>`. Chỉ giữ 5 thư mục 1 (surprise), 4 (happy), 5 (sad), 6 (angry), 7 (neutral); script báo lỗi nếu còn thư mục 2 hoặc 3. Lệnh `prepare_public_data.py` tải UTKFace về `data/`.

```powershell
.\venv\Scripts\python.exe scripts/evaluate_rafdb.py --data-dir data/rafdb/DATASET --models efficientnet --split test --output results/rafdb_test_rerun
.\venv\Scripts\python.exe scripts/prepare_public_data.py --utkface
.\venv\Scripts\python.exe scripts/evaluate_utkface_attributes.py --workers 4 --crop-scale 1.5 --output-dir results/utkface_rerun
.\venv\Scripts\python.exe scripts/calibrate_attributes.py results/utkface_rerun/predictions.csv --output results/genderage_calibration_rerun.json
.\venv\Scripts\python.exe scripts/evaluate_tracking.py
.\venv\Scripts\python.exe scripts/evaluate_quality.py
.\venv\Scripts\python.exe scripts/benchmark_pipeline.py
.\venv\Scripts\python.exe scripts/summarize_final.py
```

Train lại trên RAF-DB: `train_rafdb.py --data-dir data/rafdb/DATASET --evaluate-test` → `evaluate_rafdb.py`. Muốn đo kiểu cắt ảnh như app: `prepare_rafdb.py` rồi `evaluate_rafdb.py --manifest ...`.

## Nguồn và giấy phép

Chỉ dùng cho nghiên cứu và học tập. Không phân phối dataset hoặc ảnh mặt kèm dự án.

| Thành phần | Nguồn | Lưu ý |
| --- | --- | --- |
| MediaPipe Face Landmarker | Google | |
| `genderage.onnx` | InsightFace `buffalo_l` | Trọng số chỉ cho nghiên cứu phi thương mại |
| UTKFace | Bản sao `nu-delta/utkface` | Chỉ nghiên cứu phi thương mại |
| RAF-DB | Bản nhóm có sẵn (`RAF-DB.zip`); dùng 11.273 ảnh train, 2.834 ảnh test của 5 lớp | Chỉ nghiên cứu phi thương mại theo điều khoản của tác giả RAF-DB |
| DeepAR SDK và các filter `.deepar` | DeepAR (tải SDK từ jsDelivr lúc chạy) | Cần License Key theo domain; watermark tùy gói license |
| `efficientnet.keras` | Fine-tune 5 lớp từ checkpoint nhóm bàn giao | Chưa rõ dữ liệu train của checkpoint ban đầu |
