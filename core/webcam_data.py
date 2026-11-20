"""Shared webcam manifest, subject-disjoint splitting and image I/O."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = (ROOT / "models" / "labels.txt").read_text(encoding="utf-8").split()
# Thứ tự 7 lớp của RAF-DB (thư mục 1..7) và của đầu ra efficientnet.keras.
RAF_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
CROP_VERSION = "square-affine-v1"
DEFAULT_CROP_MARGIN = 0.0


def configure_console():
    """Vietnamese CLI output must work with Windows redirected stdout/stderr too."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')


def read_image(path):
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f'Không đọc được ảnh: {path}') from exc
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Không mã hóa được ảnh: {path}")
    encoded.tofile(str(path))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hash(path):
    """Hash decoded pixels, catching losslessly re-encoded copies as well as renamed files."""
    image = read_image(path)
    return hashlib.sha256(np.asarray(image.shape, dtype=np.int64).tobytes() + image.tobytes()).hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        official_raf = row.get('dataset') == 'rafdb' and row.get('split_basis') == 'official-rafdb'
        has_subject = row.get('subject') is not None and bool(str(row['subject']).strip())
        if row.get('label') not in LABELS or not (has_subject or official_raf):
            raise ValueError('Mỗi mẫu cần nhãn hợp lệ và subject; RAF-DB phải có split chính thức.')
        if official_raf and row.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('Mẫu RAF-DB thiếu split hợp lệ.')
        for key in ("path", "frame_path"):
            if row.get(key):
                row[key] = (path.parent / row[key]).resolve()
                if not row[key].is_file():
                    raise ValueError(f"Thiếu ảnh {row[key]}")
        if not row.get("path") or row["path"] in seen:
            raise ValueError("Thiếu path hoặc ảnh bị trùng trong manifest.")
        seen.add(row["path"])
        rows.append(row)
    if not rows:
        raise ValueError("Manifest chưa có mẫu.")
    return rows


def split_manifest(rows, val_fraction=0.15, test_fraction=0.2, seed=42):
    """Honor official RAF splits, or split webcam subjects; never fabricate person IDs."""
    explicit = [row.get('split') is not None for row in rows]
    if not any(explicit):
        return split_subjects(rows, val_fraction, test_fraction, seed)
    if not all(explicit):
        raise ValueError('Không trộn manifest có split rõ ràng với mẫu chưa có split.')
    splits = {name: [] for name in ('train', 'validation', 'test')}
    owners = {}
    for row in rows:
        split = row['split']
        if split not in splits:
            raise ValueError(f'Split không hợp lệ: {split}')
        if row.get('subject') is not None:
            subject = str(row['subject'])
            if subject in owners and owners[subject] != split:
                raise ValueError('Cùng người xuất hiện ở nhiều tập dữ liệu.')
            owners[subject] = split
        splits[split].append(row)
    if any({row['label'] for row in splits[name]} != set(LABELS) for name in ('train', 'validation')):
        raise ValueError('Train/validation phải có đủ 7 lớp.')
    return splits


def split_subjects(rows, val_fraction=0.2, test_fraction=0.2, seed=42):
    """Split whole people, searching deterministic candidates for all seven classes."""
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Tỉ lệ val/test không hợp lệ.")
    groups = sorted({str(row["subject"]) for row in rows})
    nval = max(1, round(len(groups) * val_fraction))
    ntest = max(1, round(len(groups) * test_fraction)) if test_fraction else 0
    if len(groups) <= nval + ntest:
        raise ValueError("Cần thêm người để tách train/validation/test độc lập (ít nhất 3 khi có test).")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        shuffled = rng.permutation(groups)
        selected = {"validation": set(shuffled[:nval]),
                    "test": set(shuffled[nval:nval + ntest]),
                    "train": set(shuffled[nval + ntest:])}
        splits = {name: [row for row in rows if str(row["subject"]) in subjects]
                  for name, subjects in selected.items()}
        if all({row["label"] for row in values} == set(LABELS)
               for values in splits.values() if values):
            return splits
    raise ValueError("Không tách được đủ 7 lớp theo người. Thu đủ 7 biểu cảm ở nhiều người hơn.")


def check_content_leakage(splits):
    """Reject broken images, pixel-identical cross-split copies and conflicting labels."""
    owners = {}
    for name, samples in splits.items():
        for path, label in samples:
            digest = image_hash(path)
            if digest in owners:
                previous_split, previous_label = owners[digest]
                if previous_label != label:
                    raise ValueError(f'Ảnh trùng nội dung nhưng nhãn mâu thuẫn: {path}')
                if previous_split != name:
                    raise ValueError(f'Ảnh trùng nội dung giữa {previous_split} và {name}: {path}')
            owners[digest] = (name, label)


def quality(face, box, min_size=80, min_blur=35.0):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Resize before blur measurement so camera resolution does not dominate.
    gray = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reason = ("face too small" if min(box[2:]) < min_size else
              "too dark / too bright" if not 35 <= brightness <= 225 else
              "blurry" if blur < min_blur else "ok")
    return {"brightness": brightness, "sharpness": blur, "reason": reason}
