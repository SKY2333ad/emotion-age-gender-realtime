# Nhận diện cảm xúc, tuổi và giới tính thời gian thực

App webcam nhận diện **cảm xúc** (7 lớp), **nhóm tuổi** và **giới tính ước lượng** cho từng khuôn mặt, kèm theo dõi ID. Có 3 bộ tìm mặt (Haar, MediaPipe, YOLO26) và 3 model cảm xúc train trên RAF-DB (CNN, MobileNetV3-Small, EfficientNet-B0), đổi được ngay trên giao diện. Chạy cục bộ trên CPU, không gửi ảnh ra ngoài.

Nhóm 1: Nguyễn Văn Quân (dữ liệu, train, đánh giá), Lý Thế Vinh (detector, pipeline, benchmark, kiểm thử), Nguyễn Phan Anh Tiến và Trịnh Bá Khánh Toàn (giao diện, đóng gói).

## Chạy app

**Cần:** Windows, [Python 3.12 hoặc 3.13](https://www.python.org/downloads/) 64-bit (có `py`), webcam, Internet lần đầu, khoảng 3 GB trống.

```powershell
.\setup.bat      # chỉ chạy lần đầu, cài vào venv, mất khoảng 10–15 phút
.\run_app.bat    # mở app ở http://localhost:8501
```

Vào tab **Live**, chọn bộ tìm mặt và model ở thanh bên, bấm **Bắt đầu**.

| Gặp lỗi | Cách xử lý |
| --- | --- |
| `KHONG tao duoc moi truong` | Cài Python 3.12 hoặc 3.13. Python 3.14 trở lên chưa dùng được với TensorFlow |
| `Không mở được webcam` | Đóng ứng dụng khác đang dùng camera, kiểm tra quyền camera của Windows |
| Muốn cài lại từ đầu | Xóa thư mục `venv`, rồi chạy lại `.\setup.bat` |

Kiểm tra code (75 test, khoảng 1 phút):

```powershell
.\venv\Scripts\python.exe scripts/system/check_sync.py
.\venv\Scripts\python.exe -m unittest discover -s tests
```

## Cấu trúc dự án: mỗi thư mục là một bước

```text
data/             BƯỚC 1  Dữ liệu (DB) để train/đánh giá: rafdb/, utkface/, webcam/  (không kèm ảnh, mỗi thư mục có README)
scripts/data/             Chuẩn bị DB: chia tập, kiểm tra rò rỉ, tải UTKFace
scripts/emotion/  BƯỚC 2  Cảm xúc: train 3 model trên RAF-DB, đánh giá trên tập test, xuất ONNX
scripts/age_gender/       Tuổi, giới tính: đánh giá và hiệu chuẩn model có sẵn trên UTKFace
models/           BƯỚC 3  Kết quả train: emotion/, age_gender/, face_detection/
core/             BƯỚC 4  Lõi app dùng khi chạy: tìm mặt → cắt mặt → theo dõi ID → cảm xúc → tuổi/giới tính
app.py                    Giao diện Streamlit: Live, Thống kê phiên, Kết quả thí nghiệm
scripts/webcam/   BƯỚC 5  Kiểm thử bằng webcam thật có nhãn (theo điều kiện), quay video benchmark
scripts/system/           Đo tốc độ 9 tổ hợp, mô phỏng theo dõi ID, kiểm tra hợp đồng model
results/          BƯỚC 6  Số liệu đo được: emotion/, age_gender/, webcam/, system/
tests/                    Test tự động
```

| Nhiệm vụ | Dữ liệu | Model | Script | Kết quả |
| --- | --- | --- | --- | --- |
| Tìm mặt | Không train (model có sẵn); đánh giá bằng video tự quay | `models/face_detection/`: `face_landmarker.task`, `yolo26n-face.onnx`; Haar có sẵn trong OpenCV | `scripts/system/benchmark_pipeline.py --video`, `scripts/webcam/` | `results/system/benchmark_9_to_hop/`, `results/webcam/` |
| Cảm xúc | `data/rafdb/` | `models/emotion/`: `cnn.keras`, `mobilenetv3.keras`, `efficientnet.keras` (+ `.onnx`) | `scripts/data/make_manifest.py` → `scripts/emotion/train_rafdb.py` → `scripts/emotion/evaluate_rafdb.py` | `results/emotion/` |
| Tuổi, giới tính | `data/utkface/` | `models/age_gender/genderage.onnx` (InsightFace, không train lại) + `genderage_calibration.json` | `scripts/age_gender/` | `results/age_gender/` |

| File trong `core/` | Việc |
| --- | --- |
| `pipeline.py` | Ghép mọi bước, đọc camera ở luồng riêng |
| `detectors.py` | 3 detector Haar, MediaPipe, YOLO26; cắt và căn chỉnh mặt |
| `tracker.py` | Giữ ID qua các khung |
| `quality.py` | Điểm chất lượng mặt, làm mượt theo thời gian |
| `predict.py` | Tìm model đã train, dự đoán cảm xúc (mọi mặt trong một lần gọi model) |
| `attributes.py` | Tuổi, giới tính, hiệu chuẩn |
| `fusion.py` | Hiệu chỉnh cảm xúc bằng blendshape (thử nghiệm, mặc định tắt) |
| `ui_helpers.py` | Nhóm tuổi, vẽ khung, dòng thống kê |
| `preprocessing.py`, `selection.py`, `webcam_data.py` | Tiền xử lý, luật chọn checkpoint, đọc ghi dữ liệu |

## Cách hệ thống hoạt động

```text
Webcam → luôn lấy khung mới nhất
       → Tìm mặt: Haar | MediaPipe | YOLO26 (khung thu về rộng 480 px)
       → Cắt mặt vuông, lề 15%; MediaPipe xoay cho hai mắt nằm ngang
       → Gán ID, giữ ID 10 khung khi mặt tạm mất
       → Chấm điểm chất lượng mặt
       → Cảm xúc: CNN | MobileNetV3-Small | EfficientNet-B0 → làm mượt theo chất lượng
       → Tuổi, giới tính: mỗi 5 khung → hiệu chuẩn → gộp 12 lần gần nhất
       → Vẽ lên màn hình; tab Thống kê ghi mỗi mặt 1 dòng mỗi 0,5 giây
```

Haar và YOLO26 chỉ trả khung mặt, nên với hai detector này không có căn chỉnh theo mắt và không có blendshape. Thiếu model tuổi/giới tính thì cảm xúc vẫn chạy. EfficientNet-B0 chạy bằng ONNX (nhanh khoảng gấp đôi Keras, cùng kết quả); thiếu hoặc sai file ONNX thì tự dùng Keras.

## Kết quả

### Cảm xúc trên RAF-DB

| Model | Báo cáo, 3068 ảnh test: Accuracy / Macro-F1 | File trong dự án, 674 ảnh test: Accuracy / Macro-F1 |
| --- | --- | --- |
| EfficientNet-B0 | 79,8% / 0,713 | 72,3% / 0,715 |
| CNN | 77,0% / 0,667 | 68,6% / 0,671 |
| MobileNetV3-Small | 73,3% / 0,640 | 65,9% / 0,652 |

Cột 674 ảnh (tối đa 100 ảnh mỗi lớp) lấy từ `results/emotion/rafdb_674/` của Người 1. Số trên đủ 3068 ảnh nằm trong báo cáo nhưng chưa có file gốc trong dự án; khi có RAF-DB, chạy `evaluate_rafdb.py --split test` để có lại. Ở cả ba model, happy tốt nhất, fear và disgust yếu nhất.

### Tuổi và giới tính trên UTKFace

18.780 ảnh test, CI 95% bootstrap theo ảnh: giới tính đúng 85,4% (84,8–85,9), nhóm tuổi đúng 47,4% (46,7–48,1), sai tuổi trung bình 10,27 năm (10,14–10,41). Xác suất giới tính đã hiệu chuẩn, ECE từ 0,094 xuống 0,009. Giới tính kém với trẻ em và thiếu niên nam: bé trai 0–12 tuổi đúng 48,6%, nam 13–19 đúng 52,9%.

### Tốc độ (Intel i5-10210U, chỉ CPU)

- 9 tổ hợp trên cùng video, 3 lần lặp, ngày 27/10/2026 (`results/system/benchmark_9_to_hop/2026-10-27/`): chỉ CNN gần realtime, khoảng 13,8 FPS với Haar hoặc MediaPipe. EfficientNet-B0 chạy bằng Keras chỉ đạt 2–3 FPS. YOLO26 tìm mặt chậm hơn khoảng 5 lần (86 ms/khung). Bản đo này dùng Keras cho EfficientNet; app hiện chạy ONNX nên nhanh hơn, cần đo lại.
- Pipeline đầy đủ của app (MediaPipe + EfficientNet-B0 ONNX + tuổi/giới tính), đo 12/11/2026: khoảng 27 FPS với 1 mặt và 10 FPS với 3 mặt, chưa tính camera và Streamlit (`results/system/benchmark_pipeline.json`). Máy lúc đo chạy không đều, hai lần đo cùng bản có thể lệch tới 35%.

### Webcam thật

Đo ngày 12/11/2026 (`results/webcam/conditions_log.csv`): 1 người, 7 biểu cảm, chính diện. MediaPipe + EfficientNet-B0 (lúc quay có bật hiệu chỉnh blendshape) đúng 51,4% (macro-F1 0,43); fear đúng 0% (thành surprise) và happy đúng 0% (thành disgust/sad). Chưa quay CNN và MobileNetV3-Small. Đo độ bền 9 tổ hợp theo ánh sáng, khoảng cách, góc, che khuất (27/10/2026) nằm trong `results/webcam/do_ben_9_to_hop/`; MediaPipe + EfficientNet-B0 bền nhất (trung bình 60,3%).

**Cần biết trước khi demo:** số liệu webcam mới có 1 người, chưa đủ để kết luận model nào tốt hơn. Tuổi và giới tính chỉ là ước lượng từ hình dáng khuôn mặt, không xác định bản dạng giới, không dùng để ra quyết định về ai.

## Chạy lại từng bước

### Cảm xúc (cần RAF-DB, xem `data/rafdb/README.md`)

```powershell
.\venv\Scripts\python.exe scripts/data/make_manifest.py --img_root data/rafdb/basic --train_csv data/rafdb/train_labels.csv --test_csv data/rafdb/test_labels.csv --out results/emotion/rafdb_split
.\venv\Scripts\python.exe scripts/emotion/train_rafdb.py --data-dir data/rafdb/basic --split-manifest results/emotion/rafdb_split/split_manifest.csv --dry-run
.\venv\Scripts\python.exe scripts/emotion/train_rafdb.py --data-dir data/rafdb/basic --split-manifest results/emotion/rafdb_split/split_manifest.csv --arch cnn --output models/emotion/cnn_s42.keras --evaluate-test
.\venv\Scripts\python.exe scripts/emotion/evaluate_rafdb.py --data-dir data/rafdb/basic --models cnn mobilenetv3 efficientnet --split test --output results/emotion/rafdb_test
```

- `make_manifest.py` chia tập như báo cáo: val 10% phân tầng, seed 42, ảnh trùng luôn cùng tập, ảnh train trùng ảnh test bị loại (11019 / 1224 / 3068).
- `train_rafdb.py --arch cnn|mobilenetv3|efficientnet` dựng đúng kiến trúc của 3 model trong `models/emotion/` và train theo cấu hình báo cáo: Adam 1e-3, tối đa 60 epoch, batch 64, EarlyStopping theo validation, trọng số lớp clip(N / (K × n_c), 0,5, 3,0), tăng cường lật, xoay ±0,08, zoom 0,1, dịch 0,08. MobileNetV3-Small và EfficientNet-B0 bắt đầu từ trọng số ImageNet (Keras tự tải lần đầu). Báo cáo đề xuất 3 seed: đổi `--seed 42/43/44` và `--output`.
- Ba file `cnn.keras`, `mobilenetv3.keras`, `efficientnet.keras` hiện có là bản Người 1 đã train; `train_rafdb.py` không ghi đè chúng, model mới tự hiện trong app nhờ file `.metadata.json`.
- Sau khi train lại EfficientNet-B0, xuất ONNX theo hướng dẫn đầu `scripts/emotion/export_onnx.py` rồi chạy `scripts/emotion/check_onnx.py`.

### Tuổi và giới tính (cần UTKFace, tải tự động khoảng 1 GB)

```powershell
.\venv\Scripts\python.exe scripts/data/prepare_public_data.py --utkface
.\venv\Scripts\python.exe scripts/age_gender/evaluate_utkface_attributes.py --workers 4 --crop-scale 1.5 --output-dir results/age_gender/utkface_baseline_rerun
.\venv\Scripts\python.exe scripts/age_gender/calibrate_age_groups.py results/age_gender/utkface_baseline_rerun/predictions.csv
.\venv\Scripts\python.exe scripts/age_gender/calibrate_attributes.py results/age_gender/utkface_baseline_rerun/predictions.csv --output models/age_gender/genderage_calibration_rerun.json
.\venv\Scripts\python.exe scripts/age_gender/select_attribute_config.py
.\venv\Scripts\python.exe scripts/age_gender/summarize_utkface.py
```

Validation (SHA-256 tên file % 5 == 0) dùng để chọn ngưỡng nhóm tuổi và hiệu chuẩn; test chỉ để báo cáo.

### Kiểm thử webcam và tốc độ (không cần dataset)

```powershell
.\venv\Scripts\python.exe scripts/webcam/run_conditions.py --who nguoi01 --groups A --model cnn
.\venv\Scripts\python.exe scripts/webcam/summarize_conditions.py
.\venv\Scripts\python.exe scripts/webcam/record_video.py --seconds 15
.\venv\Scripts\python.exe scripts/system/benchmark_pipeline.py --video data/webcam/bench_input.mp4
.\venv\Scripts\python.exe scripts/system/benchmark_pipeline.py
.\venv\Scripts\python.exe scripts/system/evaluate_tracking.py
```

- `run_conditions.py` nhóm A quay đủ 7 biểu cảm với cả 3 detector (21 lượt), nhóm B đổi ánh sáng, nhóm E đổi khoảng cách. Chạy lại với `--model mobilenetv3`, `--model efficientnet`, mỗi model từ 3 người trở lên. Chỉ ghi số liệu, không lưu ảnh.
- `benchmark_pipeline.py --video` đo 9 tổ hợp như mục 3 báo cáo, ghi vào `results/system/benchmark_9_to_hop/<ngày>/`.

## YOLO26

Ultralytics không phát hành YOLO26 cho khuôn mặt. Dự án dùng bản nano cộng đồng train trên WIDER FACE, [Shubham12864/YOLO26n-face](https://huggingface.co/Shubham12864/YOLO26n-face) (AGPL-3.0), giống bản báo cáo đã dùng (imgsz 640, conf 0,4). App đọc bản ONNX bằng ONNX Runtime nên không phải cài PyTorch; chuyển một lần trong môi trường riêng:

```powershell
py -3.12 -m venv E:\_yolo
E:\_yolo\Scripts\python.exe -m pip install ultralytics huggingface_hub
E:\_yolo\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; from ultralytics import YOLO; YOLO(hf_hub_download('Shubham12864/YOLO26n-face', 'yolo26_widerdataset.pt')).export(format='onnx', imgsz=640)"
```

Chép file `.onnx` vừa tạo thành `models/face_detection/yolo26n-face.onnx` (đã có sẵn trong dự án, xuất ngày 12/11/2026 bằng ultralytics 8.4.211, SHA-256 bắt đầu bằng `f5a5f1b8`). Checkpoint này tắt nhánh end-to-end, nên đầu ra là `(1, 5, 8400)` và `core/detectors.py` tự lọc trùng bằng NMS như ultralytics (conf 0,4, IoU 0,7, tối đa 5 mặt); dạng end-to-end `(1, 300, 6)` cũng đọc được. Trên ảnh mẫu, khung của bản ONNX lệch vài pixel so với ultralytics chạy file `.pt`; mỗi khung mất khoảng 110 ms trên i5-10210U.

## Kết quả lấy ở đâu

Tab **Kết quả thí nghiệm** chỉ đọc file trong `results/`, và dưới mỗi bảng ghi file cùng script tạo ra nó.

| Thư mục trong `results/` | Nội dung | Script tạo ra |
| --- | --- | --- |
| `emotion/rafdb_674/` | So sánh 3 model, F1 từng lớp, ma trận nhầm lẫn (674 ảnh test) | Script đánh giá cũ của Người 1; bản mới là `emotion/evaluate_rafdb.py` |
| `emotion/rafdb_split/`, `emotion/training/` | Cách chia tập, kiểm tra rò rỉ; log từng lần train | `data/make_manifest.py`, `emotion/train_rafdb.py` (tạo khi có RAF-DB) |
| `age_gender/utkface_baseline/`, `utkface_tta/` | Dự đoán từng ảnh UTKFace | `age_gender/evaluate_utkface_attributes.py` |
| `age_gender/attribute_calibration/`, `attribute_selection.json`, `utkface_summary.json` | Hiệu chuẩn, chọn cấu hình, CI 95% | `age_gender/calibrate_attributes.py`, `select_attribute_config.py`, `summarize_utkface.py` |
| `webcam/conditions_log.csv` và các bảng, hình cạnh nó | Webcam có nhãn, tổng hợp theo model và biểu cảm | `webcam/condition_test.py`, `run_conditions.py`, `summarize_conditions.py` |
| `webcam/do_ben_9_to_hop/` | Độ bền 9 tổ hợp theo điều kiện, 27/10/2026 | Script `run_condition_tests.py` của nhóm, không còn trong dự án |
| `system/benchmark_9_to_hop/<ngày>/` | 9 tổ hợp, mỗi lần lặp một CSV | `system/benchmark_pipeline.py --video` |
| `system/benchmark_pipeline.json`, `webcam_smoke.json` | Pipeline đầy đủ, từng model; camera thật | `system/benchmark_pipeline.py`, `--camera 12` |
| `system/tracking_evaluation.json`, `quality_evaluation.json`, `final_tests.log` | Theo dõi ID mô phỏng, điểm chất lượng, log test | `system/evaluate_tracking.py`, `evaluate_quality.py`, `unittest` |

## Ngoài phạm vi báo cáo

- **Hiệu chỉnh cảm xúc bằng blendshape** (`models/emotion/efficientnet_fusion.json`): báo cáo mục 7 quyết định không đưa vào hệ thống chính, nên chỉ là nút bật/tắt trong app, mặc định tắt, dùng được với EfficientNet-B0 và MediaPipe. Số đo trên RAF-DB của bộ này (84,7%) không dùng để so sánh được: validation của nó tách từ tập train RAF-DB, mà EfficientNet-B0 nhiều khả năng đã học trên chính tập này.
- **Đã bỏ:** model FER+ và HSEmotion cùng các script đánh giá trên FER+ (báo cáo đã chuyển từ FER2013 sang RAF-DB, và HSEmotion là model bên ngoài). Bản cũ còn trong lịch sử git (commit `a8793b6`).

## Nguồn và giấy phép

Chỉ dùng cho nghiên cứu và học tập. Không phân phối dataset hoặc ảnh mặt kèm dự án.

| Thành phần | Nguồn | Lưu ý |
| --- | --- | --- |
| RAF-DB | Nhóm tác giả RAF-DB | Phải xin quyền, chỉ nghiên cứu phi thương mại |
| `cnn.keras`, `mobilenetv3.keras`, `efficientnet.keras` | Nhóm tự train trên RAF-DB | MobileNetV3-Small và EfficientNet-B0 khởi tạo từ ImageNet |
| Haar Cascade `haarcascade_frontalface_default.xml` | OpenCV | Đi kèm opencv-python |
| MediaPipe Face Landmarker | Google | |
| `yolo26n-face.onnx` | Hugging Face `Shubham12864/YOLO26n-face`, Ultralytics YOLO26 | AGPL-3.0 |
| `genderage.onnx` | InsightFace `buffalo_l` | Trọng số chỉ cho nghiên cứu phi thương mại |
| UTKFace | Bản sao `nu-delta/utkface` | Chỉ nghiên cứu phi thương mại |
