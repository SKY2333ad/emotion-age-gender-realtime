"""
app.py - Giao diện Streamlit cho hệ thống nhận diện biểu hiện khuôn mặt


"""

import time
import json
from datetime import datetime
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detectors import DETECTOR_NAMES, DETECTORS, YOLO_MODEL
from pipeline import CameraThread, Pipeline
from predict import DEFAULT_MODEL, MODEL_NAMES, MODEL_TITLES, REPORT_MODELS, get_crop_config
from fusion import DEFAULT_PATH as fusion_file
from preprocessing import prepare_face
from ui_helpers import LABELS, draw_results, result_row
from ui_helpers import AGE_GROUPS, GENDER_NAMES, age_group, detail_text


# ---------------------------------------------------------------- Cấu hình (sửa ở đây khi nhóm chốt)
# 3 model của báo cáo đứng trước; model khác (ngoài báo cáo) ghi rõ là thử nghiệm.
MODEL_OPTIONS = {MODEL_TITLES.get(name, f'{name} (thử nghiệm)'): name for name in MODEL_NAMES}
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MAX_ROWS = 5000                                # giới hạn số dòng thống kê trong một phiên
RECORD_INTERVAL = 0.5                          # giây giữa hai dòng của cùng một mặt: 5000 dòng ≈ 40 phút với 1 mặt
UI_INTERVAL = 0.25                             # giây giữa hai lần cập nhật chữ trạng thái/FPS

st.set_page_config(page_title="Nhận diện biểu hiện khuôn mặt", page_icon="🙂", layout="wide")
st.title("Hệ thống nhận diện biểu hiện khuôn mặt")

with st.sidebar:
    st.header("Cấu hình")
    detector = st.selectbox('Phát hiện khuôn mặt', DETECTORS, index=DETECTORS.index('mediapipe'),
                            format_func=DETECTOR_NAMES.get)
    if detector == 'yolo' and not Path(YOLO_MODEL).exists():
        st.error('Chưa có models/yolo26n-face.onnx. Xem mục YOLO26 trong README.md.')
    if detector != 'mediapipe':
        st.caption('Haar và YOLO26 chỉ cho khung mặt: không căn chỉnh theo mắt, không có kết hợp blendshape.')
    missing = [MODEL_TITLES[name] for name in REPORT_MODELS if name not in MODEL_NAMES]
    if missing:
        st.caption('Chưa train: ' + ', '.join(missing) + ' (scripts/train_rafdb.py --arch).')
    model_label = st.selectbox("Mô hình", list(MODEL_OPTIONS),
                               index=list(MODEL_OPTIONS.values()).index(DEFAULT_MODEL))
    model_name = MODEL_OPTIONS[model_label]
    trained_crop = get_crop_config(model_name)
    align_faces = st.toggle("Căn chỉnh khuôn mặt", value=bool(trained_crop['align']), key='align_' + model_name)
    crop_margin = st.slider("Lề quanh khuôn mặt", 0.0, 0.5, float(trained_crop['margin']), 0.05,
                            key='margin_' + model_name,
                            help="So với ảnh lúc train và chọn lề bằng validation.")
    preview_crop = st.checkbox("Xem ảnh khuôn mặt đưa vào mô hình", value=False)
    use_attributes = st.checkbox('Nhận diện tuổi và giới tính', value=True)
    st.caption('Tuổi và giới tính là ước lượng từ hình dáng khuôn mặt, có thể sai, đặc biệt kém với trẻ em. '
               'Tuổi được hiển thị theo nhóm; giới tính dự đoán không xác định bản dạng giới.')
    fusion_config = None
    if fusion_file.exists():
        try:
            fusion_config = json.loads(fusion_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            st.error('Không đọc được bộ kết hợp MediaPipe.')
    fusion_compatible = bool(fusion_config and fusion_config.get('base_model') == model_name
                             and detector == 'mediapipe')        # blendshape chỉ MediaPipe mới có
    use_fusion = st.checkbox("Kết hợp đặc trưng MediaPipe", value=False, disabled=not fusion_compatible,
                             key=f'fusion_{model_name}_{detector}',
                             help="Thử nghiệm, mặc định tắt (báo cáo mục 7). Chỉ dùng với EfficientNet-B0 và MediaPipe.")
    if fusion_config is None:
        st.caption("Chưa có bộ kết hợp được huấn luyện.")
    elif not fusion_compatible and detector == 'mediapipe':
        st.caption('Bộ kết hợp hiện dùng cho checkpoint: ' + fusion_config['base_model'])
    if use_fusion:
        crop_margin = fusion_config['crop_config']['margin']
        align_faces = fusion_config['crop_config']['align']
        st.caption(f"Bộ kết hợp dùng lề {crop_margin:.2f}, căn chỉnh {'bật' if align_faces else 'tắt'}.")


# ---------------------------------------------------------------- Trạng thái phiên
if "rows" not in st.session_state:
    st.session_state.rows = []
if "live_pipe" not in st.session_state:
    st.session_state.live_pipe = Pipeline()


def record(results, source):
    """Lấy mẫu theo thời gian cho từng mặt, để thống kê phủ cả phiên dài thay vì vài phút cuối."""
    now, stamp = time.monotonic(), datetime.now().isoformat(timespec="milliseconds")
    last = st.session_state.setdefault('last_record', {})
    rows = st.session_state.rows
    for r in results:
        if now - last.get(r['track_id'], -RECORD_INTERVAL) >= RECORD_INTERVAL:
            last[r['track_id']] = now
            rows.append(result_row(r, stamp, source, detector, model_name))
    del rows[:-MAX_ROWS]


def run(pipe, frame_bgr):
    """Gọi pipeline đúng hợp đồng, trả (ảnh đã vẽ, kết quả)."""
    pipe.align = align_faces
    pipe.crop_margin = crop_margin
    pipe.fusion_path = str(fusion_file) if use_fusion else None
    pipe.attributes = use_attributes
    results = pipe.process_frame(frame_bgr, detector, model_name)
    return draw_results(frame_bgr, results), results


# ---------------------------------------------------------------- 3 tab
tab_live, tab_stats, tab_exp = st.tabs(
    ["Live", "Thống kê phiên", "Kết quả thí nghiệm"])

# ---- Tab Live: chỉ tạo nút và chỗ hiển thị. Vòng lặp chạy ở cuối file để hai tab còn lại vẫn được vẽ.
with tab_live:
    st.subheader("Nhận diện trực tiếp từ webcam")
    c1, c2 = st.columns(2)
    if c1.button("▶ Bắt đầu", type="primary"):
        st.session_state.live = True
    if c2.button("■ Dừng"):
        st.session_state.live = False
    view, status, fps_box, crop_box = st.empty(), st.empty(), st.empty(), st.empty()
    if not st.session_state.get("live", False):
        status.info("Nhấn **Bắt đầu** để chạy webcam.")


def live_loop():
    pipe = st.session_state.live_pipe
    pipe.reset()
    st.session_state.last_record = {}
    camera = CameraThread(0)
    if not camera.ok:
        st.session_state.live = False
        status.error("Không mở được webcam. Kiểm tra quyền camera hoặc đóng ứng dụng khác đang dùng camera.")
        return
    status.info("Đang chạy. Đổi cấu hình ở thanh bên sẽ khởi động lại vòng lặp. Nhấn Dừng để kết thúc.")
    n, t0, seen, last_frame, last_ui = 0, time.perf_counter(), 0, time.perf_counter(), 0.0
    try:
        while st.session_state.get("live", False):
            if camera.count == seen:                       # luôn xử lý khung mới nhất, không xếp hàng
                if time.perf_counter() - last_frame > 3:
                    status.warning("Không đọc được khung hình từ webcam.")
                    break
                time.sleep(0.002)
                continue
            seen, last_frame = camera.count, time.perf_counter()
            frame = cv2.flip(camera.read(), 1)               # lật gương cho tự nhiên
            try:
                started = time.perf_counter()
                annotated, results = run(pipe, frame)
                latency = (time.perf_counter() - started) * 1000
            except Exception as exc:
                status.error(f"Lỗi pipeline: {exc}")
                break
            record(results, "webcam")
            view.image(annotated, channels="BGR")
            n += 1
            now = time.perf_counter()
            if now - last_ui < UI_INTERVAL:               # chữ trạng thái chỉ cần đổi vài lần mỗi giây
                continue
            last_ui = now
            largest = max(results, key=lambda r: r['box'][2] * r['box'][3]) if results else None
            if largest:
                mode = 'EfficientNet + blendshape' if largest['fusion_used'] else model_label
                summary = mode + ' · Top 3: ' + ' · '.join(f'{k}: {v:.0%}' for k, v in largest['top3'].items())
                summary += ' · ' + detail_text(largest)
            else:
                summary = 'Không phát hiện khuôn mặt.'
            if pipe.attribute_error:
                status.warning(summary + ' · ' + pipe.attribute_error)
            else:
                status.info(summary)
            if preview_crop and largest:
                crop = prepare_face(pipe.get_face_crop(frame, largest["box"])).astype(np.uint8)
                crop_box.image(crop, caption="Ảnh RGB 100×100 đưa vào mô hình (mặt lớn nhất)", width=220)
            elif preview_crop:
                crop_box.empty()
            fps_box.metric("FPS / trễ xử lý", f"{n / max(now - t0, 1e-6):.1f} / {latency:.0f} ms")
    finally:
        camera.stop()


# ---- Tab Thống kê phiên
with tab_stats:
    st.subheader("Thống kê phiên hiện tại")
    if st.session_state.get("live", False):
        st.caption("Đang chạy webcam: số liệu dưới đây là lúc bắt đầu, nhấn Dừng để cập nhật.")
    if not st.session_state.rows:
        st.info("Chưa có dữ liệu. Hãy chạy webcam ở tab Live.")
    else:
        df = pd.DataFrame(st.session_state.rows)
        k1, k2 = st.columns(2)
        k1.metric("Số lượt ghi nhận", len(df), help=f"Mỗi mặt ghi tối đa 1 dòng mỗi {RECORD_INTERVAL:g} giây.")
        k2.metric("Nhãn hay gặp nhất", df["label"].value_counts().idxmax())
        left, right = st.columns(2)
        with left:
            st.caption("Tỉ lệ từng cảm xúc trong phiên")
            share = df["label"].value_counts(normalize=True).reindex(LABELS).fillna(0) * 100
            st.bar_chart(share)
        with right:
            st.caption("Xác suất từng cảm xúc theo thời gian")
            cols = [f"prob_{l}" for l in LABELS if f"prob_{l}" in df]
            if cols:
                line = df[["timestamp"] + cols].copy()
                line["timestamp"] = pd.to_datetime(line["timestamp"], errors="coerce")
                line = line.set_index("timestamp").rename(columns=lambda c: c[5:])
                st.line_chart(line)
        st.caption('Thống kê tuổi/giới tính tính theo lượt mặt được ghi nhận, không phải số người riêng biệt. '
                   'Đây là ước lượng từ hình dáng khuôn mặt, có thể sai; bỏ qua lượt chưa có ước lượng.')
        attr_left, attr_right = st.columns(2)
        with attr_left:
            st.caption('Tỉ lệ nam/nữ ước lượng trong phiên (%)')
            genders = df['gender'].dropna() if 'gender' in df else pd.Series(dtype=str)
            genders = genders[genders.isin(GENDER_NAMES)]
            if len(genders):
                gender_share = genders.value_counts(normalize=True).reindex(GENDER_NAMES).fillna(0) * 100
                st.bar_chart(gender_share.rename(index=GENDER_NAMES))
            else:
                st.info('Chưa có ước lượng giới tính.')
        with attr_right:
            st.caption('Phân bố nhóm tuổi ước lượng (lượt ghi nhận)')
            groups = df['age'].map(age_group).dropna() if 'age' in df else pd.Series(dtype=str)
            if len(groups):
                st.bar_chart(groups.value_counts().reindex(AGE_GROUPS).fillna(0))
            else:
                st.info('Chưa có ước lượng tuổi.')
        display_df = df.copy()
        if 'age' in display_df:
            display_df['age'] = display_df['age'].map(age_group)
        if 'gender' in display_df:
            display_df['gender'] = display_df['gender'].map(GENDER_NAMES)
        display_df = display_df.rename(columns={'age': 'Nhóm tuổi ước lượng', 'gender': 'Giới tính ước lượng',
                                                 'gender_prob': 'Xác suất giới tính'})
        st.dataframe(display_df.iloc[::-1])
        st.download_button("⬇ Tải báo cáo phiên (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="session_statistics.csv", mime="text/csv")
        if st.button("Xóa dữ liệu phiên"):
            st.session_state.rows = []
            st.rerun()


# ---- Tab Kết quả thí nghiệm: mỗi bảng ghi rõ file số liệu và script tạo ra nó
def load_result(name):
    try:
        return json.loads((RESULTS_DIR / name).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def source(files, script):
    st.caption(f"Nguồn: `results/{files}` · tạo bởi `scripts/{script}`")


def show_images(paths):
    cols = st.columns(2)
    for i, path in enumerate(paths):
        cols[i % 2].image(str(path), caption=path.name)


with tab_exp:
    st.subheader("Kết quả thí nghiệm")
    st.caption("Mỗi mục ghi file số liệu và script tạo ra nó; thiếu mục nào thì chạy script đó (lệnh trong README).")

    st.markdown("#### 1. Cảm xúc: 3 model trên RAF-DB")
    rafdb = next((folder for folder in (RESULTS_DIR / 'emotion' / 'rafdb_test', RESULTS_DIR / 'emotion' / 'rafdb_674')
                  if (folder / 'comparison.csv').exists()), None)
    if rafdb:
        st.dataframe(pd.read_csv(rafdb / 'comparison.csv'), hide_index=True)
        if rafdb.name == 'rafdb_674':
            st.caption('674 ảnh test RAF-DB (tối đa 100 ảnh mỗi lớp), file của Người 1. Báo cáo ghi số trên đủ 3068 ảnh; '
                       'chạy lại evaluate_rafdb.py để có file của bản đầy đủ.')
        with st.expander('Precision, Recall, F1 từng lớp'):
            for path in sorted(rafdb.glob('metrics_*.csv')):
                st.caption(path.stem.removeprefix('metrics_'))
                st.dataframe(pd.read_csv(path), hide_index=True)
        source(f'emotion/{rafdb.name}/', 'emotion/evaluate_rafdb.py')
        show_images(sorted(rafdb.glob('confusion_*.png')))
    else:
        st.info('Chưa có. Đặt RAF-DB vào data/rafdb rồi chạy scripts/emotion/evaluate_rafdb.py.')

    utk = (load_result('age_gender/utkface_summary.json') or {}).get('utkface_test')
    st.markdown("#### 2. Tuổi và giới tính (UTKFace test)")
    if utk:
        names = {'gender_accuracy': 'Giới tính đúng', 'age_group_accuracy': 'Nhóm tuổi đúng',
                 'age_group_macro_f1': 'Macro-F1 nhóm tuổi', 'age_mae': 'Sai số tuổi trung bình (năm)'}
        st.dataframe(pd.DataFrame([{'Chỉ số': label, 'Giá trị': round(utk[key], 3),
                                    'CI 95%': '{:.3f} – {:.3f}'.format(*utk['ci95'][key])}
                                   for key, label in names.items()]), hide_index=True)
        st.caption(f"{utk['images']} ảnh test, bootstrap {utk['rounds']} lần theo ảnh.")
        source('age_gender/utkface_summary.json', 'age_gender/summarize_utkface.py')
        show_images(sorted((RESULTS_DIR / 'age_gender' / 'attribute_calibration').glob('*.png')))
    else:
        st.info('Chưa có. Chạy các script trong scripts/age_gender/ theo README.')

    st.markdown("#### 3. Tốc độ: 3 detector × 3 model (chỉ CPU)")
    runs = sorted((RESULTS_DIR / 'system' / 'benchmark_9_to_hop').glob('*/benchmark_lan*.csv'))
    if runs:
        latest = runs[-1].parent
        frame = pd.concat(pd.read_csv(path) for path in sorted(latest.glob('benchmark_lan*.csv')))
        table = frame.groupby(['detector', 'model'], sort=False)[['ms_detect', 'ms_classify', 'ms_p95', 'fps']]
        st.dataframe(table.mean().round(1).join(table.std().round(1).add_suffix(' SD')).reset_index(), hide_index=True)
        st.caption(f'Ngày đo {latest.name}, trung bình {len(list(latest.glob("benchmark_lan*.csv")))} lần lặp trên cùng video.')
        source(f'system/benchmark_9_to_hop/{latest.name}/', 'system/benchmark_pipeline.py --video')
    speed = load_result('system/benchmark_pipeline.json')
    if speed:
        columns = ['config', 'mean_ms', 'p95_ms', 'fps_mean', 'cpu_percent', 'rss_mb']
        st.dataframe(pd.DataFrame(speed['rows'])[columns].round(1), hide_index=True)
        if speed.get('model_speed'):
            st.dataframe(pd.DataFrame(speed['model_speed']).T.round(2))
        st.caption('Pipeline đầy đủ của app (tracking, chất lượng, tuổi/giới tính, vẽ) với '
                   f"{speed.get('detector', 'mediapipe')} + {speed['model']}, và thời gian từng model cảm xúc.")
        source('system/benchmark_pipeline.json', 'system/benchmark_pipeline.py')

    st.markdown("#### 4. Cảm xúc trên webcam thật")
    webcam = RESULTS_DIR / 'webcam'
    if (webcam / 'metrics_overall.csv').exists():
        st.dataframe(pd.read_csv(webcam / 'metrics_overall.csv'), hide_index=True)
        if (webcam / 'conditions_by_expr.csv').exists():
            st.dataframe(pd.read_csv(webcam / 'conditions_by_expr.csv'), hide_index=True)
        source('webcam/metrics_overall.csv, conditions_by_expr.csv', 'webcam/summarize_conditions.py')
        show_images(sorted(webcam.glob('confusion_*.png')) + sorted(webcam.glob('chart_*.png')))
    else:
        st.info('Chưa đo. Quay bằng scripts/webcam/run_conditions.py rồi chạy summarize_conditions.py.')
    robust = webcam / 'do_ben_9_to_hop'
    if (robust / 'conditions_9combos.csv').exists():
        st.caption('Độ bền theo điều kiện, 9 tổ hợp trên cùng khung hình (báo cáo mục 6, quay 27/10/2026)')
        st.dataframe(pd.read_csv(robust / 'conditions_9combos.csv'), hide_index=True)
        source('webcam/do_ben_9_to_hop/', 'run_condition_tests.py của nhóm (không còn trong dự án)')
        show_images(sorted(robust.glob('*.png')))

    tracking = load_result('system/tracking_evaluation.json')
    if tracking:
        st.markdown("#### 5. Theo dõi ID (mô phỏng)")
        st.dataframe(pd.DataFrame(tracking['tracking']).T)
        source('system/tracking_evaluation.json', 'system/evaluate_tracking.py')

    st.markdown("#### 6. Báo cáo chi tiết")
    report_files = sorted(RESULTS_DIR.rglob('report.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if report_files:
        selected_report = st.selectbox('Báo cáo huấn luyện / đánh giá', report_files,
                                       format_func=lambda p: str(p.parent.relative_to(RESULTS_DIR)))
        try:
            report_data = json.loads(selected_report.read_text(encoding='utf-8'))
            comparison_file = selected_report.parent / 'comparison.csv'
            if comparison_file.exists():
                st.dataframe(pd.read_csv(comparison_file), hide_index=True)
            elif 'baseline_validation' in report_data:
                stages = [('Baseline', report_data['baseline_validation'])]
                if 'best_validation' in report_data:
                    stages.append(('Fine-tune', report_data['best_validation']))
                st.dataframe(pd.DataFrame([{
                    'Model': name, 'Tập': 'validation', 'Accuracy': scores['accuracy'],
                    'Macro-F1': scores['macro_f1'], 'Sad F1': scores['per_class']['sad']['f1'],
                    'Fear F1': scores['per_class']['fear']['f1']
                } for name, scores in stages]), hide_index=True)
                st.caption(f"Trạng thái: {report_data.get('status', 'unknown')} · Epoch được chọn: {report_data.get('best_epoch', 0)}")
                if report_data.get('best_epoch') == 0:
                    st.caption('Chưa chọn được checkpoint cải thiện; trọng số baseline được giữ lại.')
            per_class_file = selected_report.parent / 'metrics_per_class.csv'
            if per_class_file.exists():
                st.dataframe(pd.read_csv(per_class_file), hide_index=True)
            with st.expander('Chi tiết báo cáo'):
                st.json(report_data)
            show_images(sorted(selected_report.parent.glob('*.png')))
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f'Không đọc được báo cáo: {exc}')
    else:
        st.info('Chưa có report.json nào trong results/.')


if st.session_state.get("live", False):
    live_loop()
