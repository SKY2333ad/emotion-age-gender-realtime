"""Tải và giải nén dữ liệu đánh giá công khai, ghim đúng phiên bản Hugging Face.

UTKFace: ảnh cho scripts/evaluate_utkface_attributes.py (chỉ nghiên cứu phi thương mại).
FER+: 7 lớp, 48×48, không có ID người và không có split gốc; chia ổn định theo hash nội dung ảnh.
python scripts/prepare_public_data.py --utkface --ferplus
"""
import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
from webcam_data import configure_console

UTKFACE = ('nu-delta/utkface', 'ecceea6bbfaed3c9dd287e9cc034d0c6bfcd82d5',
           [f'data/train-0000{index}-of-00003.parquet' for index in range(3)])
FERPLUS = ('deanngkl/ferplus-7cls', 'a057d0768d39864227130033d3dfaaa3e056f0b5',
           ['data/train-00000-of-00001.parquet'])
FER_TO_PROJECT = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def fetch(repo, revision, name, directory):
    target = directory / Path(name).name
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.partial')
        urllib.request.urlretrieve(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}', temporary)
        temporary.replace(target)
    return target


def prepare_utkface(downloads, output):
    repo, revision, names = UTKFACE
    images = output / 'UTKFace'
    images.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in names:
        path = fetch(repo, revision, name, downloads / 'utkface')
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=256, columns=['image', 'file_name']):
            for image, file_name in zip(batch['image'].to_pylist(), batch['file_name'].to_pylist()):
                safe = Path(file_name).name               # tên file đến từ dữ liệu tải về, không tin đường dẫn
                (images / safe).write_bytes(image['bytes'])
                count += 1
    (output / 'source.json').write_text(json.dumps({
        'repo': repo, 'revision': revision, 'images': count,
        'license': 'UTKFace: chỉ nghiên cứu phi thương mại; nhãn tuổi/giới tính ước lượng rồi kiểm tra thủ công.',
        'race_labels': 'Do người tạo dataset gán, không phải tự khai; không dùng để kết luận công bằng.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('UTKFace:', count, images)


def prepare_ferplus(downloads, output):
    repo, revision, names = FERPLUS
    path = fetch(repo, revision, names[0], downloads / 'ferplus')
    pixels, labels = [], []
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=1024, columns=['image', 'label']):
        for image, label in zip(batch['image'].to_pylist(), batch['label'].to_pylist()):
            pixels.append(np.asarray(Image.open(io.BytesIO(image['bytes'])).convert('L').resize((48, 48))))
            labels.append(FER_TO_PROJECT[label])
    pixels = np.stack(pixels).astype(np.uint8)
    digests = [hashlib.sha256(item.tobytes()).digest() for item in pixels]
    groups = {digest: index for index, digest in enumerate(dict.fromkeys(digests))}
    # 20% validation, 80% test theo hash nội dung: ảnh trùng pixel luôn ở cùng một tập.
    split = np.array(['validation' if digest[0] % 5 == 0 else 'test' for digest in digests])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'ferplus.npz', pixels=pixels, labels=np.array(labels), split=split,
                        group=np.array([groups[digest] for digest in digests]))
    (output / 'source.json').write_text(json.dumps({
        'repo': repo, 'revision': revision, 'images': len(pixels), 'unique_pixels': len(groups),
        'split': {name: int((split == name).sum()) for name in ('validation', 'test')},
        'license': 'Bản sao bên thứ ba, không ghi giấy phép; chưa xác minh với nguồn gốc FER+. Chỉ dùng đánh giá nghiên cứu, không phân phối lại.',
        'limitations': 'Ảnh xám 48×48, nguồn web; không có ID người.'}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FER+:', len(pixels), output)


def near_duplicates(output):
    """Dò ảnh gần trùng giữa validation và test bằng average hash 8×8; chỉ báo cáo, không đổi cách chia tập."""
    data = np.load(output / 'ferplus.npz')
    owners = {}
    for item, split, group in zip(data['pixels'], data['split'], data['group']):
        small = np.asarray(Image.fromarray(item).resize((8, 8), Image.BOX), dtype=float)
        owners.setdefault((small > small.mean()).tobytes(), []).append((str(split), int(group)))
    crossing = [items for items in owners.values() if len({name for name, _ in items}) > 1 and len({group for _, group in items}) > 1]
    report = {'method': 'average hash 8×8, không phân biệt được ảnh khác người có bố cục giống nhau',
              'hash_groups': len(owners), 'groups_crossing_splits': len(crossing),
              'images_in_crossing_groups': sum(len(items) for items in crossing)}
    (output / 'near_duplicates.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--utkface', action='store_true')
    parser.add_argument('--ferplus', action='store_true')
    parser.add_argument('--near-duplicates', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    downloads = args.data_dir / '_downloads'
    if args.utkface:
        prepare_utkface(downloads, args.data_dir / 'utkface')
    if args.ferplus:
        prepare_ferplus(downloads, args.data_dir / 'ferplus')
    if args.near_duplicates:
        near_duplicates(args.data_dir / 'ferplus')


if __name__ == '__main__':
    main()
