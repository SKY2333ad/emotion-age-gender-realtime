"""Local web server for Photo Booth.

Serves the static interface and exposes POST /api/analyze. Images are processed
in memory by the project's existing MediaPipe + EfficientNet pipeline and are
never written to disk.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


SITE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SITE_DIR.parents[1]
CORE_DIR = PROJECT_DIR / "core"
sys.path.insert(0, str(CORE_DIR))

from pipeline import Pipeline  # noqa: E402
from predict import DEFAULT_MODEL, LABELS  # noqa: E402
from quality import face_quality  # noqa: E402
from webcam_data import DEFAULT_CROP_MARGIN  # noqa: E402


PIPELINE = Pipeline(
    align=True,
    crop_margin=DEFAULT_CROP_MARGIN,
    attributes=True,
    quality_gate=True,
)
PIPELINE_LOCK = threading.Lock()


STATUS_MESSAGES = {
    "OK": "Đầu vào tốt",
    "NEED_ALIGNMENT": "Đã tự căn chỉnh khuôn mặt",
    "LARGE_POSE": "Hãy nhìn thẳng vào camera",
    "TOO_FAR": "Hãy đến gần camera hơn",
    "LOW_LIGHT": "Hãy tăng ánh sáng phía trước",
    "BLURRY": "Giữ khuôn mặt ổn định",
    "LOW_CONFIDENCE": "AI chưa đủ chắc chắn",
}


def _decode_image(data_url: str) -> np.ndarray:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Thiếu dữ liệu ảnh hợp lệ.")
    encoded = data_url.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Ảnh vượt quá giới hạn 8 MB.")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Không đọc được ảnh.")
    return frame


def _number(value):
    return None if value is None else float(value)


def analyze(frame: np.ndarray) -> dict:
    started = time.perf_counter()
    with PIPELINE_LOCK:
        results = PIPELINE.process_frame(frame, "mediapipe", DEFAULT_MODEL)
        if results:
            result = max(results, key=lambda item: item["box"][2] * item["box"][3])
            face_meta = next((face for face in PIPELINE._faces if face.get("box") == result["box"]), {})
    if not results:
        return {"ok": True, "detected": False, "probs": {label: 0.0 for label in LABELS},
                "latencyMs": (time.perf_counter() - started) * 1000,
                "message": "Không tìm thấy khuôn mặt"}

    x, y, width, height = result["box"]
    image_h, image_w = frame.shape[:2]
    quality = face_quality(frame, result["box"], face_meta.get("angle", 0.0), face_meta.get("yaw", 0.0))
    status = result.get("status") or "OK"
    label = result.get("label")

    return {
        "ok": True,
        "detected": True,
        "ready": label is not None and status not in {"LARGE_POSE", "TOO_FAR", "LOW_LIGHT", "BLURRY"},
        "emotion": label,
        "probability": _number(result.get("prob")) or 0.0,
        "probs": {emotion: float(result.get("probs", {}).get(emotion, 0.0)) for emotion in LABELS},
        "top3": {key: float(value) for key, value in result.get("top3", {}).items()},
        "gender": result.get("gender"),
        "genderProbability": _number(result.get("gender_prob")),
        "quality": {
            "light": float(quality["lighting"]),
            "sharp": float(quality["sharpness"]),
            "size": float(quality["size"]),
            "angle": float(quality["pose"]),
            "overall": float(quality["score"]),
        },
        "box": {
            "x": x / image_w,
            "y": y / image_h,
            "width": width / image_w,
            "height": height / image_h,
        },
        "status": status,
        "message": STATUS_MESSAGES.get(status, "Đang phân tích"),
        "model": DEFAULT_MODEL,
        "latencyMs": (time.perf_counter() - started) * 1000,
    }


class PhotoBoothHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Registry Windows có thể gán .mjs = text/plain; trình duyệt sẽ từ chối nạp module MediaPipe.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm"}

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._json({
                "ok": True,
                "model": DEFAULT_MODEL,
                "attributes": True,
            })
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/session/start":
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                self.rfile.read(length)
            with PIPELINE_LOCK:
                PIPELINE.reset()
            self._json({"ok": True, "reset": True})
            return
        if self.path != "/api/analyze":
            self._json({"ok": False, "error": "Không tìm thấy API."}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 12 * 1024 * 1024:
                raise ValueError("Dữ liệu gửi lên không hợp lệ.")
            payload = json.loads(self.rfile.read(length))
            self._json(analyze(_decode_image(payload.get("image", ""))))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # Keep the live UI responsive while reporting inference errors.
            self._json({"ok": False, "error": f"Lỗi AI: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, message: str, *args) -> None:
        print(f"[PhotoBooth] {self.address_string()} - {message % args}")


def main() -> None:
    host, port = "127.0.0.1", 4173
    handler = partial(PhotoBoothHandler, directory=str(SITE_DIR / "dist"))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Photo Booth: http://localhost:{port}")
    print("Images are processed in memory and are not stored by the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
