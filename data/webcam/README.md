# Video webcam tự quay

Đầu vào chung cho benchmark 9 tổ hợp detector × model (mục 3 báo cáo), để mọi tổ hợp chạy trên đúng cùng các khung hình.

```powershell
.\venv\Scripts\python.exe scripts/webcam/record_video.py --seconds 15
.\venv\Scripts\python.exe scripts/system/benchmark_pipeline.py --video data/webcam/bench_input.mp4
```

Video có mặt người thật: chỉ quay người đã đồng ý và không chia sẻ ra ngoài nhóm.
