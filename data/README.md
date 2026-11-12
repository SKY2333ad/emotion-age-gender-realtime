# data/: dữ liệu để train và đánh giá

Ảnh và video không kèm theo dự án (dung lượng, giấy phép, ảnh mặt người thật). Mỗi thư mục con có README ghi cần đặt gì, lấy ở đâu và script nào dùng nó.

| Thư mục | Dùng cho | Script đọc |
| --- | --- | --- |
| `rafdb/` | Train và đánh giá 3 model cảm xúc (CNN, MobileNetV3-Small, EfficientNet-B0) | `scripts/data/make_manifest.py`, `scripts/data/prepare_rafdb.py`, `scripts/emotion/*` |
| `utkface/` | Hiệu chuẩn và đánh giá tuổi, giới tính (`genderage.onnx`, không train lại) | `scripts/data/prepare_public_data.py`, `scripts/age_gender/*` |
| `webcam/` | Video tự quay làm đầu vào benchmark 9 tổ hợp | `scripts/webcam/record_video.py`, `scripts/system/benchmark_pipeline.py --video` |

Phát hiện mặt (Haar, MediaPipe, YOLO26) dùng model đã train sẵn nên không có thư mục dữ liệu train riêng. Log quay webcam có nhãn chỉ chứa số liệu, không có ảnh, nên nằm trong `results/webcam/`.
