"""E1/E2/E4 và mục 7.4: thử loss, cân bằng lớp, MLP và điều chỉnh quyết định trên đầu ra EfficientNet.

Chỉ train lại lớp cuối trên đặc trưng đã trích (CPU đủ chạy), không train lại backbone.
Train: phần FER+ chưa dùng trong evaluate_ferplus. Chọn: validation. Test: chỉ cấu hình thắng, một lần.
python scripts/experiment_emotion_head.py --output results/emotion_head
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
from predict import LABELS, _get
from preprocessing import prepare_face
from scripts.evaluate_ferplus import bootstrap, calibration, confusion, pick
from scripts.train_rafdb import scores_from_confusion
from webcam_data import CROP_VERSION, configure_console, file_hash

SEED, EPOCHS, BATCH = 42, 40, 128


def extract(keras, pixels, batch_size=64):
    """Đầu ra GlobalAveragePooling của mạng đã train, cùng tiền xử lý với app; chạy graph như predict.py."""
    import tensorflow as tf
    model = _get('efficientnet')
    network = keras.Model(model.input, model.get_layer('global_average_pooling2d').output)
    extractor = tf.function(lambda images: network(images, training=False),
                            input_signature=[tf.TensorSpec((None, 100, 100, 3), tf.float32)], autograph=False)
    output = []
    for offset in range(0, len(pixels), batch_size):
        batch = np.stack([prepare_face(cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)) for item in pixels[offset:offset + batch_size]])
        output.append(extractor(batch).numpy())
    return np.concatenate(output)


def build(keras, kind, dimension):
    inputs = keras.Input((dimension,))
    hidden = inputs
    if kind == 'mlp':
        hidden = keras.layers.Dropout(0.3)(keras.layers.Dense(128, activation='relu')(hidden))
    return keras.Model(inputs, keras.layers.Dense(7, activation='softmax')(hidden))


def train(keras, config, features, targets, val_features, val_targets):
    keras.utils.set_random_seed(SEED)
    model = build(keras, config.get('head', 'linear'), features.shape[1])
    rng = np.random.default_rng(SEED)
    x, y = features, targets
    if config.get('oversample'):
        groups = [np.flatnonzero(targets == label) for label in range(7)]
        index = np.concatenate([rng.choice(group, max(map(len, groups))) for group in groups])
        x, y = features[index], targets[index]
    one_hot = np.eye(7)[y]
    if config.get('mixup'):
        order, mix = rng.permutation(len(x)), rng.beta(.2, .2, (len(x), 1))
        x, one_hot = mix * x + (1 - mix) * x[order], mix * one_hot + (1 - mix) * one_hot[order]
    counts = np.bincount(targets, minlength=7)
    weight = {index: float(len(targets) / (7 * counts[index])) for index in range(7)} if config.get('weighted') else None
    loss = (keras.losses.CategoricalFocalCrossentropy(alpha=1.0, gamma=2.0) if config.get('focal')
            else keras.losses.CategoricalCrossentropy(label_smoothing=config.get('smoothing', 0.0)))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    best, best_weights = -1.0, None
    for _ in range(EPOCHS):
        model.fit(x, one_hot, batch_size=BATCH, epochs=1, class_weight=weight, verbose=0, shuffle=True)
        score = macro(val_targets, model(val_features, training=False).numpy())
        if score > best:
            best, best_weights = score, model.get_weights()
    model.set_weights(best_weights)
    return model


def macro(truth, probs):
    return scores_from_confusion(confusion(truth, probs.argmax(axis=1)))['macro_f1']


def fit_bias(probs, truth, rounds=3):
    """Mục 7.4: cộng thêm một hằng số vào log-xác suất từng lớp, dò từng lớp theo macro-F1 validation."""
    logits, bias = np.log(np.clip(probs, 1e-9, 1)), np.zeros(7)
    for _ in range(rounds):
        for label in range(7):
            scores = []
            for value in np.arange(-2, 2.01, 0.1):
                trial = bias.copy()
                trial[label] = value
                scores.append((macro(truth, logits + trial), value))
            bias[label] = max(scores)[1]
    return bias


def summary(name, truth, probs):
    scores = scores_from_confusion(confusion(truth, probs.argmax(axis=1)))
    return {'config': name, 'macro_f1': scores['macro_f1'], 'accuracy': scores['accuracy'],
            **{label: scores['per_class'][label]['f1'] for label in ('fear', 'sad', 'disgust')},
            'neutral_to_sad': scores['confusion_matrix'][LABELS.index('neutral')][LABELS.index('sad')]}


def export_model(keras, head, args, report):
    """Ghép mạng gốc với lớp cuối mới thành checkpoint riêng; app tự nhận qua metadata, baseline không đổi."""
    target = ROOT / 'models' / 'efficientnet_ferplus.keras'
    if target.exists():
        print('Giữ nguyên checkpoint đã có:', target)
        return
    base = _get('efficientnet')
    features = base.get_layer('global_average_pooling2d').output
    model = keras.Model(base.input, head(features))
    model.save(target)
    target.with_suffix('.metadata.json').write_text(json.dumps({
        'sha256': file_hash(target), 'crop_config': {'margin': 0.15, 'align': True, 'version': CROP_VERSION},
        'labels': LABELS, 'status': 'complete', 'selection_passed': True, 'experimental': True,
        'initial_sha256': file_hash(ROOT / 'models' / 'efficientnet.keras'), 'best_epoch': None,
        'trained_on': 'FER+ (xám 48×48), chỉ lớp cuối; không phải dữ liệu webcam',
        'report': str(args.output.relative_to(ROOT) / 'report.json'),
        'selected_on_validation': report['selected_on_validation']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Đã xuất:', target)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'ferplus' / 'ferplus.npz')
    parser.add_argument('--limit', type=int, default=3000)
    parser.add_argument('--train-images', type=int, default=15000)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'emotion_head')
    args = parser.parse_args()
    from tensorflow import keras
    data = np.load(args.data)
    truth = np.array([LABELS.index(label) for label in data['labels']])
    validation = pick(data['split'], 'validation', args.limit)
    test = pick(data['split'], 'test', args.limit * 7 // 3)
    fit = np.setdiff1d(np.flatnonzero(data['split'] == 'test'), test)
    # Loại ảnh trùng pixel với mẫu test, để train không chứa bản sao của ảnh test.
    fit = fit[~np.isin(data['group'][fit], data['group'][test])]
    args.output.mkdir(parents=True, exist_ok=True)
    def features(name, index):
        cache = args.output / f'features_{name}.npy'
        if not cache.exists():
            np.save(cache, extract(keras, data['pixels'][index]))
            print('Đã trích', name, len(index), flush=True)
        return np.load(cache)

    fit = fit[np.random.default_rng(SEED).permutation(len(fit))[:args.train_images]]
    fit.sort()
    train_x, val_x, test_x = features('train', fit), features('validation', validation), features('test', test)
    train_y, val_y, test_y = truth[fit], truth[validation], truth[test]
    model = _get('efficientnet')
    head = model.get_layer('dense')
    original = lambda x: head(x).numpy()      # lớp cuối gốc trên cùng đặc trưng (dropout không chạy khi suy luận)
    results = [summary('E0 lớp cuối gốc', val_y, original(val_x))]
    configs = {'E1a cross entropy': {}, 'E1b focal': {'focal': True}, 'E1c label smoothing 0.1': {'smoothing': .1},
               'E2a weighted CE': {'weighted': True}, 'E2b oversample': {'oversample': True},
               'E2c MixUp': {'mixup': True}, 'E4 MLP nhỏ': {'head': 'mlp'}}
    models = {}
    for name, config in configs.items():
        models[name] = train(keras, config, train_x, train_y, val_x, val_y)
        results.append(summary(name, val_y, models[name](val_x, training=False).numpy()))
        print(results[-1], flush=True)
    # File FER+ xếp theo lớp nên phải trộn trước khi chia đôi.
    shuffled = np.random.default_rng(SEED).permutation(len(val_y))
    first, second = shuffled[:len(val_y) // 2], shuffled[len(val_y) // 2:]
    bias = fit_bias(original(val_x[first]), val_y[first])
    adjusted = np.log(np.clip(original(val_x[second]), 1e-9, 1)) + bias
    results.append(dict(summary('7.4 bias theo lớp (fit nửa này, đo nửa kia)', val_y[second], adjusted),
                        reference_macro_f1=macro(val_y[second], original(val_x[second]))))
    print(results[-1], flush=True)
    best_name = max(configs, key=lambda name: next(row['macro_f1'] for row in results if row['config'] == name))
    probs = models[best_name](test_x, training=False).numpy()
    base = original(test_x)
    report = {'train_images': len(fit), 'validation_images': len(validation), 'test_images': len(test), 'seed': SEED,
              'epochs': EPOCHS, 'validation_results': results, 'selected_on_validation': best_name,
              'test_baseline': summary('E0 lớp cuối gốc', test_y, base), 'test_selected': summary(best_name, test_y, probs),
              'test_selected_ci95': bootstrap(test_y, probs.argmax(axis=1), data['group'][test]),
              'test_baseline_ci95': bootstrap(test_y, base.argmax(axis=1), data['group'][test]),
              'test_calibration_selected': calibration(probs, test_y)}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    models[best_name].save(args.output / 'best_head.keras')
    export_model(keras, models[best_name], args, report)
    print('Chọn:', best_name, report['test_selected'], 'gốc:', report['test_baseline'])


if __name__ == '__main__':
    main()
