"""Luật chọn checkpoint khi fine-tune: macro-F1 phải tăng, sad và angry không được giảm."""
import numpy as np


def qualifies(scores, baseline, best_macro_f1, max_other_drop=0.03):
    labels = list(baseline['per_class'])
    values = [scores['macro_f1'], best_macro_f1]
    values += [scores['per_class'][label]['f1'] for label in labels]
    values += [baseline['per_class'][label]['f1'] for label in labels]
    if not np.isfinite(values).all():
        return False
    if scores['macro_f1'] <= best_macro_f1 + 1e-8:
        return False
    if any(scores['per_class'][label]['f1'] < baseline['per_class'][label]['f1'] for label in ('sad', 'angry')):
        return False
    others = [label for label in labels if label not in ('sad', 'angry')]
    return float(np.mean([baseline['per_class'][label]['f1'] - scores['per_class'][label]['f1']
                          for label in others])) <= max_other_drop
