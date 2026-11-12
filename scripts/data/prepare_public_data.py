"""Bước 1 của tuổi/giới tính: tải UTKFace về data/utkface/UTKFace, ghim đúng phiên bản Hugging Face.

Dùng cho scripts/age_gender/ (hiệu chuẩn và đánh giá genderage.onnx); chỉ nghiên cứu phi thương mại.
python scripts/data/prepare_public_data.py --utkface
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'core'))
from webcam_data import configure_console

UTKFACE = ('nu-delta/utkface', 'ecceea6bbfaed3c9dd287e9cc034d0c6bfcd82d5',
           [f'data/train-0000{index}-of-00003.parquet' for index in range(3)])


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


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--utkface', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    downloads = args.data_dir / '_downloads'
    if args.utkface:
        prepare_utkface(downloads, args.data_dir / 'utkface')


if __name__ == '__main__':
    main()
