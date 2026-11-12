# RAF-DB (basic)

Bộ dữ liệu train và đánh giá 3 model cảm xúc. Phải xin quyền từ nhóm tác giả RAF-DB và chỉ dùng cho nghiên cứu phi thương mại, nên dự án không kèm ảnh.

Đặt như sau (giống bản gốc của nhóm tác giả):

```text
data/rafdb/
  basic/Image/aligned/train_00001_aligned.jpg ...   ảnh căn chỉnh 100×100
  basic/EmoLabel/list_patition_label.txt            nhãn gốc 1..7
  train_labels.csv, test_labels.csv                 cột image,label (bản Kaggle có sẵn), cho make_manifest.py
```

Nhãn số: 1 surprise, 2 fear, 3 disgust, 4 happy, 5 sad, 6 angry, 7 neutral. Bản dạng thư mục `train/1..7`, `test/1..7` cũng dùng được với `train_rafdb.py --data-dir`.

Các bước, chạy từ thư mục dự án:

1. `scripts/data/make_manifest.py`: chia train/val/test như báo cáo (val 10% phân tầng, seed 42, loại ảnh train trùng ảnh test) và tạo `results/emotion/rafdb_split/` (`split_manifest.csv`, `class_distribution.csv`, `leak_report.csv`, `split_summary.json`).
2. `scripts/emotion/train_rafdb.py --arch cnn|mobilenetv3|efficientnet --split-manifest ...`: train.
3. `scripts/emotion/evaluate_rafdb.py --split test`: đánh giá một lần trên 3068 ảnh test.
