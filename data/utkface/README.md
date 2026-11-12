# UTKFace

Dùng để hiệu chuẩn và đánh giá ước lượng tuổi, giới tính. Dự án không train lại model tuổi/giới tính: `models/age_gender/genderage.onnx` (InsightFace `buffalo_l`) đã train sẵn; UTKFace dùng để chọn ngưỡng nhóm tuổi, hiệu chuẩn xác suất giới tính (validation) và báo cáo kết quả (test).

Tải tự động (khoảng 1 GB, bản sao `nu-delta/utkface` trên Hugging Face, chỉ nghiên cứu phi thương mại):

```powershell
.\venv\Scripts\python.exe scripts/data/prepare_public_data.py --utkface
```

Ảnh nằm ở `data/utkface/UTKFace/<tuổi>_<giới>_<chủng tộc>_<thời gian>.jpg`. Chia validation/test cố định theo SHA-256 tên file (`% 5 == 0` là validation), xem `scripts/age_gender/calibrate_age_groups.py`.
