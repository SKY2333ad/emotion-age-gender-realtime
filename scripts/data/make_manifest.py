# -*- coding: utf-8 -*-
"""
make_manifest.py - Tạo MINH CHỨNG cho phần dữ liệu (góp ý câu 2):
  - split_manifest.csv      : từng ảnh thuộc tập nào (train / val / test / excluded)
  - class_distribution.csv  : số mẫu từng lớp theo từng tập
  - split_summary.json      : seed, tỉ lệ, số liệu rò rỉ, mã băm (SHA-256) của manifest
  - augmentation_config.json: cấu hình tăng cường dữ liệu + huấn luyện (khớp train.py)
  - leak_report.csv         : ảnh trùng/gần trùng giữa train và test (đã loại khỏi train)

Hai kiểu dữ liệu đầu vào:
  (A) Thư mục theo lớp (FER2013):  <data_dir>/train/<lớp>/*.jpg  và  <data_dir>/test/<lớp>/*.jpg
  (B) File nhãn CSV (RAF-DB: cột image,label; label 1..7):
        --img_root <thư mục chứa ảnh, tìm đệ quy> --train_csv train_labels.csv --test_csv test_labels.csv
Nhãn số của RAF-DB: 1 surprise, 2 fear, 3 disgust, 4 happiness, 5 sadness, 6 anger, 7 neutral.

Cách chạy:
    python make_manifest.py --data_dir data/fer2013 --out evidence_fer2013
    python make_manifest.py --img_root <thư mục RAF-DB> --train_csv <train_labels.csv> --test_csv <test_labels.csv> --out evidence_rafdb

Nguyên tắc chống rò rỉ (data leakage):
  1. Tập test = tập test chính thức, được giữ nguyên.
  2. Validation chỉ tách từ train, phân tầng theo lớp, seed cố định.
  3. Ảnh trùng/gần trùng (dHash) luôn nằm CÙNG một tập (không bị tách giữa train và val).
  4. Ảnh train trùng/gần trùng với ảnh test bị loại khỏi train.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

RAF_IDS = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happiness",
           "5": "sadness", "6": "anger", "7": "neutral"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def class_name(folder_name):
    return RAF_IDS.get(folder_name, folder_name)


def list_images(root):
    """Trả về danh sách (đường_dẫn, lớp, tập_gốc)."""
    items = []
    for split in ("train", "test"):
        sdir = root / split
        if not sdir.is_dir():
            sys.exit(f"Không thấy thư mục {sdir}")
        for cdir in sorted(d for d in sdir.iterdir() if d.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in EXTS:
                    items.append((f, class_name(cdir.name), split))
    return items


_SMALL = {}


def small_img(path):
    """Ảnh xám 32x32 (cache) để xác minh cặp ứng viên trùng lặp bằng so sánh điểm ảnh."""
    k = str(path)
    if k not in _SMALL:
        data = np.fromfile(k, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        _SMALL[k] = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _SMALL[k]


def list_from_csv(img_root, train_csv, test_csv):
    """Kiểu (B): đọc nhãn từ CSV (image,label); tìm ảnh theo tên file trong img_root."""
    index = {}
    for f in img_root.rglob("*"):
        if f.suffix.lower() in EXTS:
            index.setdefault(f.name, f)
    items, missing = [], 0
    for split, csv_path in (("train", train_csv), ("test", test_csv)):
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                p = index.get(row["image"].strip())
                if p is None:
                    missing += 1
                    continue
                items.append((p, class_name(str(row["label"]).strip()), split))
    if missing:
        print(f"Cảnh báo: {missing} dòng trong CSV không tìm thấy file ảnh trong {img_root}")
    return items


def dhash64(path):
    """dHash 64 bit: ảnh -> xám 9x8 -> so sánh điểm ảnh kề nhau."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    ap = argparse.ArgumentParser(description="Tạo manifest chia tập + thống kê + báo cáo rò rỉ")
    ap.add_argument("--data_dir", help="Kiểu (A): thư mục có train/<lớp>/ và test/<lớp>/")
    ap.add_argument("--img_root", help="Kiểu (B): thư mục chứa ảnh (tìm đệ quy)")
    ap.add_argument("--train_csv", help="Kiểu (B): file nhãn train (cột image,label)")
    ap.add_argument("--test_csv", help="Kiểu (B): file nhãn test (cột image,label)")
    ap.add_argument("--out", default="evidence")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near_dup", type=int, default=2,
                    help="Ngưỡng khoảng cách Hamming (dHash 64 bit) để chọn ứng viên trùng; -1 = tắt")
    ap.add_argument("--verify_mad", type=float, default=6.0,
                    help="Ứng viên chỉ được coi là trùng nếu sai khác điểm ảnh trung bình (0-255, ảnh 32x32) <= ngưỡng này")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.train_csv and args.test_csv and args.img_root:
        root = Path(args.img_root)
        items = list_from_csv(root, args.train_csv, args.test_csv)
    elif args.data_dir:
        root = Path(args.data_dir)
        items = list_images(root)
    else:
        sys.exit("Cần --data_dir (kiểu A) hoặc --img_root + --train_csv + --test_csv (kiểu B)")
    n = len(items)
    print(f"Tìm thấy {n} ảnh. Đang tính dHash...")
    hashes, bad = [], 0
    for i, (p, _, _) in enumerate(items):
        h = dhash64(p)
        if h is None:
            bad += 1
            h = 0
        hashes.append(h)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n}")
    if bad:
        print(f"Cảnh báo: {bad} ảnh không đọc được")
    H = np.array(hashes, dtype=np.uint64)

    # ---- gom ảnh trùng / gần trùng thành nhóm
    dsu = DSU(n)
    if args.near_dup >= 0:
        print("Đang tìm ảnh trùng/gần trùng...")
        for i in range(n - 1):
            x = np.bitwise_xor(H[i + 1:], H[i])
            d = POP[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            for j in np.nonzero(d <= args.near_dup)[0]:
                j = int(i + 1 + j)
                mad = float(np.abs(small_img(items[i][0]) - small_img(items[j][0])).mean())
                if mad <= args.verify_mad:      # xác minh bằng điểm ảnh, loại báo nhầm của dHash
                    dsu.union(i, j)
    groups = np.array([dsu.find(i) for i in range(n)])
    classes = sorted({c for _, c, _ in items})
    orig = np.array([s for _, _, s in items])
    labels = np.array([c for _, c, _ in items])

    # ---- loại ảnh train trùng với ảnh test
    status = np.array(["used"] * n, dtype=object)
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    leak_rows = []
    for g, idx in by_group.items():
        sp = {orig[i] for i in idx}
        if "train" in sp and "test" in sp:
            for i in idx:
                if orig[i] == "train":
                    status[i] = "excluded_dup_with_test"
            leak_rows.append((g, [str(items[i][0].relative_to(root)) for i in idx]))

    # ---- tách validation từ train (phân tầng theo lớp, nhóm trùng đi cùng nhau)
    from sklearn.model_selection import StratifiedGroupKFold
    tr_idx = np.array([i for i in range(n) if orig[i] == "train" and status[i] == "used"])
    k = max(2, round(1 / args.val_ratio))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed)
    _, va_local = next(iter(sgkf.split(np.zeros(len(tr_idx)), labels[tr_idx], groups[tr_idx])))
    split = np.array(["test" if o == "test" else "excluded" for o in orig], dtype=object)
    split[tr_idx] = "train"
    split[tr_idx[va_local]] = "val"

    # kiểm tra: không có nhóm nào nằm ở cả train và val
    gs = defaultdict(set)
    for i in range(n):
        if split[i] in ("train", "val"):
            gs[groups[i]].add(split[i])
    straddle = sum(1 for v in gs.values() if len(v) > 1)
    assert straddle == 0, "Có nhóm ảnh trùng bị tách giữa train và val!"

    # ---- ghi manifest
    mpath = out / "split_manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "original_split", "split", "group_id", "dhash"])
        for i, (p, c, o) in enumerate(items):
            w.writerow([p.relative_to(root).as_posix(), c, o, split[i], int(groups[i]), f"{hashes[i]:016x}"])
    sha = hashlib.sha256(mpath.read_bytes()).hexdigest()

    # ---- thống kê lớp
    cnt = {s: Counter(labels[split == s]) for s in ("train", "val", "test")}
    with open(out / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "val", "test", "total", "train_pct"])
        ttr = sum(cnt["train"].values())
        for c in classes:
            tot = cnt["train"][c] + cnt["val"][c] + cnt["test"][c]
            w.writerow([c, cnt["train"][c], cnt["val"][c], cnt["test"][c], tot,
                        f"{100 * cnt['train'][c] / max(ttr, 1):.1f}"])
        w.writerow(["TOTAL", ttr, sum(cnt["val"].values()), sum(cnt["test"].values()),
                    ttr + sum(cnt["val"].values()) + sum(cnt["test"].values()), "100.0"])

    with open(out / "leak_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "files_in_group (train bị loại, test giữ lại)"])
        for g, files in leak_rows:
            w.writerow([g, " | ".join(files)])

    conflict = sum(1 for g, idx in by_group.items()
                   if len(idx) > 1 and len({labels[i] for i in idx}) > 1)
    dup_groups_in_train = sum(1 for g, idx in by_group.items()
                              if len(idx) > 1 and {orig[i] for i in idx} == {"train"})
    import sklearn
    summary = {
        "data_root": str(root), "seed": args.seed, "val_ratio": args.val_ratio,
        "near_dup_hamming_threshold": args.near_dup,
        "counts": {s: int((split == s).sum()) for s in ("train", "val", "test", "excluded")},
        "verify_mad_threshold": args.verify_mad,
        "train_groups_with_duplicates": int(dup_groups_in_train),
        "duplicate_groups_with_conflicting_labels": int(conflict),
        "groups_spanning_train_and_test": len(leak_rows),
        "train_images_excluded_dup_with_test": int((status == "excluded_dup_with_test").sum()),
        "groups_straddling_train_val": straddle,
        "manifest_sha256": sha,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "opencv": cv2.__version__, "scikit-learn": sklearn.__version__},
    }
    (out / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- cấu hình tăng cường / huấn luyện (khớp train.py)
    aug = {
        "note": "Khớp train.py. Dùng CHUNG cho mọi classifier để so sánh công bằng.",
        "augmentation": {
            "where": "Lớp Keras nằm TRONG mô hình; chỉ chạy khi training=True, KHÔNG áp dụng cho val/test",
            "offline_augmentation_before_split": False,
            "layers": [
                {"RandomFlip": "horizontal"},
                {"RandomRotation": {"factor": 0.08, "meaning": "±0.08 x 2π ≈ ±28.8°"}},
                {"RandomZoom": 0.1},
                {"RandomTranslation": {"height": 0.08, "width": 0.08}},
            ],
        },
        "preprocessing": {"input": "100x100x3 -> Resizing 96x96", "rescale": "x/255 cố định (không dùng thống kê của dữ liệu)"},
        "class_weight": {"formula": "clip(N / (K * n_c), 0.5, 3.0)", "computed_from": "chỉ tập train"},
        "training": {"optimizer": "Adam(lr=1e-3)", "loss": "sparse_categorical_crossentropy",
                     "epochs_max": 60, "batch": 64,
                     "ReduceLROnPlateau": {"monitor": "val_loss", "factor": 0.5, "patience": 4, "min_lr": 1e-6},
                     "EarlyStopping": {"monitor": "val_accuracy", "patience": 12, "restore_best_weights": True},
                     "ModelCheckpoint": {"monitor": "val_accuracy", "save_best_only": True},
                     "model_selection_uses": "chỉ tập validation; tập test chỉ dùng 1 lần ở bước đánh giá cuối"},
    }
    (out / "augmentation_config.json").write_text(json.dumps(aug, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- in kết quả
    print("\n=== PHÂN BỐ LỚP ===")
    print(f"{'lớp':<10}{'train':>8}{'val':>8}{'test':>8}{'tổng':>8}")
    for c in classes:
        print(f"{c:<10}{cnt['train'][c]:>8}{cnt['val'][c]:>8}{cnt['test'][c]:>8}"
              f"{cnt['train'][c] + cnt['val'][c] + cnt['test'][c]:>8}")
    print("\n=== RÒ RỈ / TRÙNG LẶP ===")
    print(f"Nhóm ảnh trùng chỉ trong train          : {dup_groups_in_train}")
    print(f"Nhóm trùng vắt qua train & test          : {len(leak_rows)}")
    print(f"Ảnh train bị loại vì trùng với test      : {summary['train_images_excluded_dup_with_test']}")
    print(f"Nhóm trùng nhưng NHÃN KHÁC NHAU           : {conflict}")
    print(f"Nhóm bị tách giữa train và val (phải = 0): {straddle}")
    print(f"\nSHA-256 manifest: {sha}")
    print(f"Đã ghi 5 file vào: {out.resolve()}")


if __name__ == "__main__":
    main()