"""Hàm thuần (không phụ thuộc Streamlit) cho app.py: lọc, vẽ khung, chuẩn hóa kết quả."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from quality import STATUS_TEXT

LABELS = ["surprise", "happy", "sad", "angry", "neutral"]   # thứ tự hợp đồng, khớp models/labels.txt
GENDER_NAMES = {'male': 'nam', 'female': 'nữ'}


def attribute_text(result):
    gender = GENDER_NAMES.get(result.get('gender'))
    return f'giới tính ước lượng: {gender}' if gender else 'Chưa có ước lượng giới tính'


@lru_cache(maxsize=1)
def _attribute_font():
    # Pillow hỗ trợ dấu tiếng Việt; cv2.putText không vẽ được "nữ", "giới tính".
    for font in ('C:/Windows/Fonts/segoeui.ttf', 'DejaVuSans.ttf',
                 '/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:
            return ImageFont.truetype(font, 18)
        except OSError:
            pass
    return ImageFont.load_default()


def filter_results(results, threshold):
    """Giữ kết quả có prob >= threshold (0..1)."""
    return [r for r in (results or []) if float(r.get("prob", 0.0)) >= threshold]


def draw_results(frame_bgr, results):
    """Vẽ khung và nhãn lên bản sao của ảnh (pipeline không tự vẽ)."""
    out = frame_bgr.copy()
    attribute_lines = []
    for r in results:
        x, y, w, h = [int(v) for v in r["box"]]
        identity = f'#{r["track_id"]} ' if r.get('track_id') is not None else ''
        if r.get('label') is None:            # bị cổng chất lượng chặn: chỉ hiện mã trạng thái
            color, text = (0, 165, 255), f'{identity}{r.get("status", "")}'
        else:
            uncertain = '?' if r.get('status') == 'LOW_CONFIDENCE' else ''
            color, text = (0, 200, 0), f'{identity}{r["label"]}{uncertain} {r["prob"]:.0%}'
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, text, (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if r.get('gender') in GENDER_NAMES:
            attribute_lines.append((x, min(max(y + 2, 0), max(out.shape[0] - 24, 0)), attribute_text(r)))
    if attribute_lines:
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for x, y, text in attribute_lines:
            draw.text((max(x, 0), y), text, font=_attribute_font(), fill=(0, 220, 0),
                      stroke_width=1, stroke_fill=(0, 0, 0))
        out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return out


def detail_text(result):
    """Dòng trạng thái cho một mặt."""
    parts = [f"Mặt #{result['track_id']}", f"chất lượng {result['quality']:.0%}"]
    if result.get('status'):
        parts.append(STATUS_TEXT.get(result['status'], result['status']))
    if result.get('gender') is not None:
        parts.append(attribute_text(result))
        parts.append(f"xác suất giới tính {result['gender_prob']:.0%}")
    return ' · '.join(parts)


def result_row(r, ts, source, detector, model):
    """Một dòng thống kê: nhãn lấy từ khóa của probs (không đọc labels.txt)."""
    row = {"timestamp": ts, "source": source, "detector": detector, "model": model,
           "label": r["label"], "prob": float(r["prob"])}
    row.update({name: r.get(name) for name in ('track_id', 'quality', 'gender', 'gender_prob')})
    for k, v in (r.get("probs") or {}).items():
        row[f"prob_{k}"] = float(v)
    return row
