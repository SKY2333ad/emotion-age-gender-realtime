"""Prepare square MediaPipe-aligned RAF-DB crops and blendshapes, preserving every test image."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from detectors import detect_image_faces, align_crop, LM_MODEL
from webcam_data import LABELS, CROP_VERSION, file_hash, read_image, write_image, check_content_leakage, configure_console
from scripts.train_rafdb import load_samples, split_validation, check_splits


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 0 <= args.margin <= 0.5 or not 0 < args.val_fraction < 1:
        parser.error('margin trong [0,0.5]; val-fraction trong (0,1).')
    if args.output.exists() and not args.resume:
        parser.error('output đã tồn tại; dùng --resume để tiếp tục cùng cấu hình.')
    train, validation, test = load_samples(args.data_dir.resolve())
    if not validation:
        train, validation = split_validation(train, args.val_fraction, args.seed)
    splits = {'train': train, 'validation': validation, 'test': test}
    check_splits(splits)
    check_content_leakage(splits)
    samples = [(split, path, label) for split, values in splits.items() for path, label in values]
    signature_data = {'samples': [(split, str(path), label, file_hash(path)) for split, path, label in samples],
                      'margin': args.margin, 'seed': args.seed, 'val_fraction': args.val_fraction,
                      'crop_version': CROP_VERSION, 'landmarker_sha256': file_hash(LM_MODEL)}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / 'preparation.json'
    manifest = args.output / 'manifest.jsonl'
    partial = args.output / 'manifest.partial.jsonl'
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding='utf-8'))
        if old.get('signature') != signature:
            parser.error('Cấu hình hoặc dữ liệu nguồn đã đổi; chọn thư mục output mới.')
        if manifest.exists() and old.get('status') == 'complete':
            print(f'Dữ liệu đã chuẩn bị xong: {manifest}')
            return 0
    state = {'signature': signature, 'status': 'processing', 'source': str(args.data_dir.resolve()),
             'samples': len(samples), 'completed': 0, 'detected': 0, 'fallback': 0,
             'split_basis': 'Official RAF-DB test; validation stratified by image from train. Subject IDs unavailable.',
             'crop_config': {'margin': args.margin, 'align': True, 'version': CROP_VERSION},
             'landmarker_sha256': signature_data['landmarker_sha256']}
    completed = []
    if partial.exists():
        for line in partial.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            completed.append(row)
            if not (args.output / row['path']).is_file():
                parser.error('Thiếu crop đã ghi trong phần chuẩn bị trước.')
        if len(completed) > len(samples):
            parser.error('Manifest dở dang không hợp lệ.')
        for row, (split, path, label) in zip(completed, samples):
            if (row['split'], row['source_path'], row['label']) != (split, str(path), LABELS[label]):
                parser.error('Thứ tự dữ liệu đã đổi; dùng output mới.')
    state['completed'] = len(completed)
    state['detected'] = sum(row['landmark_detected'] for row in completed)
    state['fallback'] = len(completed) - state['detected']

    def save_state():
        temporary = state_path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(state_path)

    save_state()
    with partial.open('a', encoding='utf-8') as log:
        for index in range(len(completed), len(samples)):
            split, source, label = samples[index]
            image = read_image(source)
            faces = detect_image_faces(image)
            face = faces[0] if len(faces) == 1 else None
            box = face['box'] if face else (0, 0, image.shape[1], image.shape[0])
            angle = face['angle'] if face else 0.0
            crop = align_crop(image, box, angle, args.margin)
            crop_path = Path('crops') / split / LABELS[label] / f'{index:06d}.png'
            write_image(args.output / crop_path, crop)
            row = {'path': crop_path.as_posix(), 'frame_path': str(source), 'source_path': str(source),
                   'source_sha256': signature_data['samples'][index][3], 'label': LABELS[label],
                   'dataset': 'rafdb', 'subject': None, 'split': split, 'split_basis': 'official-rafdb',
                   'box': box, 'angle': angle, 'blendshapes': face['blendshapes'] if face else None,
                   'crop_margin': args.margin, 'align': True, 'crop_version': CROP_VERSION,
                   'landmark_detected': face is not None,
                   'landmarker_sha256': state['landmarker_sha256']}
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            state['completed'] += 1
            state['detected' if face else 'fallback'] += 1
            if state['completed'] % 250 == 0:
                save_state()
                print(f"Prepared {state['completed']}/{len(samples)}; landmarks={state['detected']}; fallback={state['fallback']}", flush=True)
    partial.rename(manifest)
    state['status'] = 'complete'
    save_state()
    print(f'Đã giữ nguyên mọi ảnh, kể cả test không có landmark. Manifest: {manifest}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
