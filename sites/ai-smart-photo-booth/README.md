# Photo Booth

## Chạy thử

Từ thư mục gốc project:

```powershell
.\run.bat booth   # hoặc nhấp đúp run.bat rồi chọn 1
```

Mở `http://localhost:4173`, chọn một trong bốn chủ đề **Ngầu / Sinh nhật / Kinh dị / Giáng sinh**, vào màn hình chụp và cấp quyền camera. FaceLandmarker chạy trực tiếp trong trình duyệt; API Python nhận ảnh JPEG rộng tối đa 480 px, chất lượng 0,7 theo vòng tuần tự, tối thiểu 250 ms giữa hai lần gửi. Vòng sau chỉ bắt đầu khi vòng trước đã hoàn tất nên không có hàng đợi request.

## Face filter

- `dist/ar/faceTracker.js`: MediaPipe FaceLandmarker ở chế độ `VIDEO`, 478 landmark, ma trận biến đổi, vòng `requestAnimationFrame` và EMA chống rung.
- `dist/ar/filterRenderer.js`: vẽ filter Canvas lên `#arCanvas` theo `chooseFilter()` (5 biểu cảm × nam/nữ = 10 filter, chủ đề chỉ đổi không khí nền; `dist/ar/filters/expressionMatrix.js`), chuyển filter bằng hiệu ứng mờ dần. Filter này luôn bật và được ghép vào ảnh chụp.
- `dist/ar/expressionFusion.js`: làm mượt xác suất EfficientNet bằng EMA; xác suất cao nhất dưới 0,35 thì coi là bình thản; nhãn mới phải cao hơn nhãn hiện tại ít nhất 0,04 và giữ 140 ms mới được đổi.
- `dist/ar/filterPolicy.js`: ánh xạ thuần từ chủ đề + biểu cảm + giới tính đáng tin cậy sang cấu hình filter; giới tính chỉ dùng khi xác suất từ 0,75. `manualProfile()` dùng cho mục **Filter**: người dùng bấm chọn 1 trong 10 filter (hình xem trước vẽ bằng chính filter đó trên một mặt hoạt hình, không cần camera), khi đó web ngừng gửi ảnh đi nhận diện và chỉ cần thấy mặt là chụp được.
- `dist/ar/ambientEffects.js`: hiệu ứng Canvas riêng cho từng chủ đề, giảm một nửa số hạt khi bật Reduce Motion.
- `dist/frameLayouts.js`: khung thành phẩm 1 ảnh lớn + 2 ảnh nhỏ, ảnh đặt kiểu `contain`.
- `dist/ar/filters/`: các module có chung giao diện `{ id, name, tags, draw(ctx, face, time) }`.
- `dist/vendor/mediapipe/` và `dist/assets/models/`: runtime WASM và model được phục vụ trên localhost, không cần CDN lúc chạy.

Biểu cảm lấy từ EfficientNet qua `/api/analyze`. Model chỉ có 5 biểu cảm (`happy`, `surprise`, `neutral`, `sad`, `angry`); mỗi biểu cảm có một filter cho nam và một cho nữ. Giới tính dưới 0,75 thì chỉ hiện viền màu theo biểu cảm. Nếu không thấy mặt hoặc đầu vào bị `LARGE_POSE`, `TOO_FAR`, `LOW_LIGHT`, `BLURRY`, canvas AR được ẩn ngay.

Không gộp với blendshape MediaPipe: trên RAF-DB test, blendshape gần như không bao giờ thấy buồn hay giận, nên gộp vào làm `sad`/`angry` gần như không bao giờ hiện. Hiệu ứng của `sad` và `angry` luôn nhẹ, vui hoặc động viên và không hiển thị nhãn tiêu cực lên người dùng.

## Kiểm tra policy

```powershell
cd sites\ai-smart-photo-booth
npm test
```

Lệnh trên chạy test `filterPolicy`, `expressionFusion`, `frameLayouts`, `ambientEffects` và `genderFilters` bằng Node test runner.

Ảnh camera và ảnh chụp chỉ nằm trong RAM của trình duyệt / tiến trình Python, không được ghi ra đĩa bởi server.
