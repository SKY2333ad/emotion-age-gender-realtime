"""Quay video ngắn từ webcam làm đầu vào chung cho benchmark 9 tổ hợp (scripts/system/benchmark_pipeline.py --video).

    python scripts/webcam/record_video.py                                  (30 giây -> data/webcam/bench_input.mp4)
    python scripts/webcam/record_video.py --seconds 15 --out data/webcam/thu.mp4

Nhìn vào camera, lần lượt làm các biểu cảm, ánh sáng tốt. Nhấn q để dừng sớm.
Video có mặt người thật nên nằm trong data/ (không kèm dự án); chỉ quay người đã đồng ý.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "core"))

import cv2
from pipeline import open_camera
from webcam_data import configure_console


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "webcam", "bench_input.mp4"))
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    cap = open_camera(args.cam)
    if cap is None:
        sys.exit("Không mở được camera (đóng ứng dụng khác đang dùng camera).")
    ok, frame = cap.read()
    if not ok:
        sys.exit("Camera không trả về khung hình.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    started, count = time.time(), 0
    print(f"Đang quay {args.seconds:.0f} giây, {width}×{height} -> {args.out}. Nhấn q để dừng sớm.")
    try:
        while time.time() - started < args.seconds:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)                 # lật gương như app
            writer.write(frame)
            count += 1
            view = frame.copy()
            cv2.putText(view, f"REC {time.time() - started:4.1f}/{args.seconds:.0f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("record (q = dung)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Xong: {count} khung -> {args.out}")


if __name__ == "__main__":
    main()
