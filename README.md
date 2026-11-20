# Phân tích khuôn mặt thời gian thực

App webcam nhận diện **5 cảm xúc** (ngạc nhiên, vui, buồn, tức giận, bình thản), **giới tính ước lượng** và **theo dõi ID** từng khuôn mặt. Chạy cục bộ trên CPU, không gửi ảnh ra ngoài. Kèm website **Photo Booth** chụp ảnh với filter theo cảm xúc. Đồ án Nhóm 1: Vinh, Tiến, Toàn, Quân.

## Chạy

**Cần:** Windows 10/11, [Python 3.12 hoặc 3.13](https://www.python.org/downloads/) 64-bit, webcam, Chrome/Edge, Internet lần đầu, khoảng 4 GB trống.

Nhấp đúp `run.bat` rồi chọn:

| Phím | Việc |
| --- | --- |
| 1 | Web Photo Booth, mở `http://127.0.0.1:4173` |
| 2 | Photo Booth qua Internet bằng ngrok (máy cần có ngrok và authtoken; thiếu thì vẫn chạy local) |
| 3 | App nhận diện realtime (Streamlit), mở `http://localhost:8501`, vào tab **Live** bấm **Bắt đầu** |
| 4 | Dừng server Photo Booth |

Lần đầu `run.bat` tự tìm Python (không có thì cài Python 3.13 bằng winget), tạo `venv`, cài `requirements.txt` (10–20 phút) rồi chạy `check_sync.py`. Các lần sau chạy ngay. Có thể gọi thẳng `run.bat booth`, `run.bat public`, `run.bat app`, `run.bat stop`.

| Gặp lỗi | Cách xử lý |
| --- | --- |
| `Can Python 3.12 hoac 3.13` | Cài Python 3.12 hoặc 3.13. Python 3.14 trở lên chưa dùng được với TensorFlow |
| `Không mở được webcam` | Đóng ứng dụng khác đang dùng camera, kiểm tra quyền camera của Windows |
| Server Photo Booth không lên | Xem `.runtime\server-error.log` |
| Muốn cài lại từ đầu | Xóa thư mục `venv`, rồi chạy lại `run.bat` |

Kiểm tra code (56 test Python, 18 test web):

```powershell
.\venv\Scripts\python.exe scripts/check_sync.py
.\venv\Scripts\python.exe -m unittest discover -s tests
cd sites\ai-smart-photo-booth; npm test
```

## Web Photo Booth

Website chụp 3 ảnh theo chủ đề (Ngầu, Sinh nhật, Kinh dị, Giáng sinh), dùng cùng pipeline cảm xúc, giới tính và cổng chất lượng ở trên qua API `/api/analyze`. Trình duyệt chạy thêm MediaPipe FaceLandmarker để vẽ filter bám mặt; cảm xúc lấy từ xác suất EfficientNet, làm mượt rồi mới đổi filter. Filter khuôn mặt vẽ bằng Canvas bám theo 478 điểm mốc, chạy sẵn không cần cài thêm: mỗi cảm xúc có một filter cho nam và một filter cho nữ (10 filter, `dist/ar/filters/expressionMatrix.js`); chủ đề chỉ đổi không khí nền (mũ tiệc, tuyết, dơi...). Giới tính chưa đủ chắc (dưới 0,75) thì chỉ hiện viền màu theo cảm xúc. Không muốn dùng nhận diện cảm xúc thì bấm chọn một trong 10 filter (có hình xem trước) ở mục **Filter** bên phải màn hình chụp: web ngừng gửi ảnh lên `/api/analyze`, chỉ còn bám mặt trong trình duyệt để vẽ filter; chọn lại "Tự động" thì nhận diện tiếp. Không cần License Key hay tài khoản dịch vụ nào; không đóng gói authtoken ngrok vào dự án.

Server chỉ nghe `127.0.0.1` và không ghi ảnh xuống đĩa. Mã nguồn web nằm trong `sites/ai-smart-photo-booth/` (chi tiết trong README của thư mục đó).

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
run.bat             menu chạy (gọi scripts/run.ps1)
app.py              giao diện Streamlit
core/               phần lõi app dùng
models/             model và tham số hiệu chuẩn
sites/ai-smart-photo-booth/   website Photo Booth (server.py + dist/)
scripts/            train, đánh giá, đo tốc độ; app không gọi
tests/              test tự động
results/            số liệu đo được
data/               dataset, không kèm dự án
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
| Train và đánh giá cảm xúc | `train_rafdb.py`, `prepare_rafdb.py`, `evaluate_rafdb.py` |
| Đánh giá giới tính | `evaluate_utkface_attributes.py`, `calibrate_attributes.py` |
| Đo pipeline | `evaluate_tracking.py`, `evaluate_quality.py`, `benchmark_pipeline.py` |
| Khác | `prepare_public_data.py`, `summarize_final.py`, `check_sync.py`, `run.ps1` |

## Việc còn lại của nhóm

**Thử với người thật trên webcam:** một mặt và hai mặt, đổi chỗ, che mặt, đổi 5 biểu cảm, thiếu sáng, quay ngang, chạy 30 phút, rút camera khi đang chạy. Toàn bộ số đo cảm xúc hiện có là trên ảnh RAF-DB (test 2.834 ảnh: accuracy 84,8%, macro-F1 0,819; F1 sad 0,82, angry 0,75 — `results/rafdb_test/`).

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

Dataset không kèm trong dự án. Cảm xúc dùng RAF-DB: đặt ảnh theo dạng `data/rafdb/DATASET/train/<cảm xúc>/` và `data/rafdb/DATASET/test/<cảm xúc>/`, với `<cảm xúc>` là `surprise`, `happy`, `sad`, `angry`, `neutral`. Bản RAF-DB gốc đánh số thư mục 1–7: đổi 1→surprise, 4→happy, 5→sad, 6→angry, 7→neutral và bỏ 2 (fear), 3 (disgust); thư mục có tên khác thì script báo lỗi. Lệnh `prepare_public_data.py` tải UTKFace về `data/`.

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
| `efficientnet.keras` | Fine-tune 5 lớp từ checkpoint nhóm bàn giao | Chưa rõ dữ liệu train của checkpoint ban đầu |
