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

from pipeline import CameraThread, Pipeline
from predict import DEFAULT_MODEL, EXPERIMENTAL_MODELS, MODEL_NAMES, get_crop_config
from preprocessing import prepare_face
from ui_helpers import LABELS, draw_results, filter_results, result_row
from ui_helpers import AGE_GROUPS, GENDER_NAMES, age_group, detail_text


# ---------------------------------------------------------------- Cấu hình (sửa ở đây khi nhóm chốt)
MODEL_OPTIONS = {'EfficientNet (baseline)': 'efficientnet'}
MODEL_OPTIONS.update({f'EfficientNet — {name.removeprefix("efficientnet_")}': name
                      for name in MODEL_NAMES if name != 'efficientnet'})
MODEL_OPTIONS.update({f'{name} (thử nghiệm)': name for name in EXPERIMENTAL_MODELS})
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MAX_ROWS = 5000                                # giới hạn số dòng thống kê trong một phiên

st.set_page_config(page_title="Nhận diện biểu hiện khuôn mặt", page_icon="🙂", layout="wide")
st.title("Hệ thống nhận diện biểu hiện khuôn mặt")

with st.sidebar:
    st.header("Cấu hình")
    detector = 'mediapipe'
    st.caption('Phát hiện khuôn mặt: MediaPipe')
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
    quality_gate = st.checkbox('Kiểm tra chất lượng đầu vào', value=True,
                               help='Không dự đoán khi mặt nghiêng quá 30°, nhỏ hơn 80 px, quá tối hoặc mờ.')
    st.caption('Tuổi và giới tính là ước lượng từ hình dáng khuôn mặt, có thể sai, đặc biệt kém với trẻ em. '
               'Tuổi được hiển thị theo nhóm; giới tính dự đoán không xác định bản dạng giới.')
    fusion_file = Path(__file__).resolve().parent / 'models' / 'efficientnet_fusion.json'
    fusion_config = None
    if fusion_file.exists():
        try:
            fusion_config = json.loads(fusion_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            st.error('Không đọc được bộ kết hợp MediaPipe.')
    fusion_compatible = bool(fusion_config and fusion_config.get('base_model') == model_name)
    use_fusion = st.checkbox("Kết hợp đặc trưng MediaPipe", value=fusion_compatible, disabled=not fusion_compatible,
                             key='fusion_' + model_name,
                             help="Chỉ khả dụng sau khi train fusion và cải thiện trên validation.")
    if fusion_config is None:
        st.caption("Chưa có bộ kết hợp được huấn luyện.")
    elif not fusion_compatible:
        st.caption('Bộ kết hợp hiện dùng cho checkpoint: ' + fusion_config['base_model'])
    if use_fusion:
        crop_margin = fusion_config['crop_config']['margin']
        align_faces = fusion_config['crop_config']['align']
        st.caption(f"Bộ kết hợp dùng lề {crop_margin:.2f}, căn chỉnh {'bật' if align_faces else 'tắt'}.")
threshold = 0.0                                # không lọc theo ngưỡng tin cậy


# ---------------------------------------------------------------- Trạng thái phiên
if "rows" not in st.session_state:
    st.session_state.rows = []
if "live_pipe" not in st.session_state:
    st.session_state.live_pipe = Pipeline()


def record(results, source):
    now = datetime.now().isoformat(timespec="milliseconds")
    rows = st.session_state.rows
    for r in results:
        if r['label'] is None:                  # khung bị cổng chất lượng chặn, không có dự đoán
            continue
        rows.append(result_row(r, now, source, detector, model_name))
    del rows[:-MAX_ROWS]


def run(pipe, frame_bgr):
    """Gọi pipeline đúng hợp đồng, lọc theo ngưỡng, trả (ảnh đã vẽ, kết quả)."""
    pipe.align = align_faces
    pipe.crop_margin = crop_margin
    pipe.fusion_path = str(fusion_file) if use_fusion else None
    pipe.attributes = use_attributes
    pipe.quality_gate = quality_gate
    results = filter_results(pipe.process_frame(frame_bgr, detector, model_name), threshold)
    return draw_results(frame_bgr, results), results


def show_results(results):
    if not results:
        st.warning("Không phát hiện khuôn mặt, hoặc không có kết quả vượt ngưỡng tin cậy.")
        return
    for i, r in enumerate(results, 1):
        st.write(f"**Khuôn mặt {i}:** {r['label']} - {r['prob']:.1%}")
        if use_attributes:
            st.caption(detail_text(r))
        probs = r.get("probs") or {}
        if probs:
            st.bar_chart(pd.Series(probs, name="Xác suất"), height=200)


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- 3 tab
tab_live, tab_stats, tab_exp = st.tabs(
    ["Live", "Thống kê phiên", "Kết quả thí nghiệm"])

# ---- Tab Live (vòng lặp OpenCV)
with tab_live:
    st.subheader("Nhận diện trực tiếp từ webcam")
    c1, c2 = st.columns(2)
    if c1.button("▶ Bắt đầu", type="primary"):
        st.session_state.live = True
    if c2.button("■ Dừng"):
        st.session_state.live = False
    view, status, fps_box, crop_box = st.empty(), st.empty(), st.empty(), st.empty()

    if st.session_state.get("live", False):
        st.session_state.live_pipe.reset()
        camera = CameraThread(0)
        if not camera.ok:
            st.session_state.live = False
            status.error("Không mở được webcam. Kiểm tra quyền camera hoặc đóng ứng dụng khác đang dùng camera.")
        else:
            status.info("Đang chạy. Đổi cấu hình ở thanh bên sẽ khởi động lại vòng lặp. Nhấn Dừng để kết thúc.")
            n, t0, seen, last_frame = 0, time.perf_counter(), 0, time.perf_counter()
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
                        annotated, results = run(st.session_state.live_pipe, frame)
                        latency = (time.perf_counter() - started) * 1000
                    except Exception as exc:
                        status.error(f"Lỗi pipeline: {exc}")
                        break
                    record(results, "webcam")
                    view.image(rgb(annotated))
                    if results:
                        largest_result = max(results, key=lambda r: r['box'][2] * r['box'][3])
                        mode = 'EfficientNet + blendshape' if largest_result['fusion_used'] else model_label
                        summary = mode + ' · Top 3: ' + ' · '.join(f'{k}: {v:.0%}' for k, v in largest_result['top3'].items())
                        summary += ' · ' + detail_text(largest_result)
                        if largest_result['label'] is None:
                            status.warning(detail_text(largest_result))
                        elif st.session_state.live_pipe.attribute_error:
                            status.warning(summary + ' · ' + st.session_state.live_pipe.attribute_error)
                        else:
                            status.info(summary)
                    elif st.session_state.live_pipe.attribute_error:
                        status.warning(st.session_state.live_pipe.attribute_error)
                    else:
                        status.info('Không phát hiện khuôn mặt.')
                    if preview_crop:
                        if results:
                            largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
                            pipe = st.session_state.live_pipe
                            crop = prepare_face(pipe.get_face_crop(frame, largest["box"])).astype(np.uint8)
                            crop_box.image(crop, caption="Ảnh RGB 100×100 đưa vào mô hình (mặt lớn nhất)", width=220)
                        else:
                            crop_box.empty()
                    n += 1
                    fps_box.metric("FPS / trễ xử lý", f"{n / max(time.perf_counter() - t0, 1e-6):.1f} / {latency:.0f} ms")
            finally:
                camera.stop()
    else:
        status.info("Nhấn **Bắt đầu** để chạy webcam.")

# ---- Tab Thống kê phiên
with tab_stats:
    st.subheader("Thống kê phiên hiện tại")
    if not st.session_state.rows:
        st.info("Chưa có dữ liệu. Hãy chạy webcam ở tab Live.")
    else:
        df = pd.DataFrame(st.session_state.rows)
        k1, k2 = st.columns(2)
        k1.metric("Số kết quả ghi nhận", len(df))
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

# ---- Tab Kết quả thí nghiệm
with tab_exp:
    st.subheader("Kết quả thí nghiệm")
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
            for plot in sorted(selected_report.parent.glob('confusion_*.png')):
                st.image(str(plot), caption=plot.stem)
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f'Không đọc được báo cáo: {exc}')
    st.caption("Đọc từ thư mục results/: benchmark.csv, conditions_summary.csv, metrics_*.csv, ảnh biểu đồ/confusion matrix.")
    RESULTS_DIR.mkdir(exist_ok=True)
    csvs = sorted({*RESULTS_DIR.glob("benchmark.csv"), *RESULTS_DIR.glob("conditions_summary.csv"),
                   *RESULTS_DIR.glob("metrics_*.csv"), *RESULTS_DIR.glob("comparison.csv")})
    if not csvs:
        st.info("Chưa có dữ liệu thí nghiệm. Đặt benchmark.csv, conditions_summary.csv, metrics_*.csv vào thư mục results/.")
    for p in csvs:
        st.markdown(f"**{p.name}**")
        try:
            st.dataframe(pd.read_csv(p))
        except Exception as exc:
            st.warning(f"Không đọc được {p.name}: {exc}")
    imgs = sorted({*RESULTS_DIR.glob("*.png"), *RESULTS_DIR.glob("*.jpg"), *RESULTS_DIR.glob("*.jpeg")})
    if imgs:
        cols = st.columns(2)
        for i, p in enumerate(imgs):
            with cols[i % 2]:
                st.image(str(p), caption=p.name)
    else:
        st.info("Chưa có ảnh biểu đồ hoặc confusion matrix trong results/.")
