# AI Smart Photo Booth AR

## Chạy thử

Từ thư mục gốc project:

```powershell
.\run_photo_booth.bat
```

Mở `http://localhost:4173`, chọn một trong bốn chủ đề **Ngầu / Sinh nhật / Kinh dị / Giáng sinh**, vào màn hình chụp và cấp quyền camera. FaceLandmarker chạy trực tiếp trong trình duyệt; API Python nhận ảnh JPEG rộng tối đa 480 px, chất lượng 0,7 theo vòng tuần tự, tối thiểu 250 ms giữa hai lần gửi. Vòng sau chỉ bắt đầu khi vòng trước đã hoàn tất nên không có hàng đợi request.

## Face filter

- `dist/ar/faceTracker.js`: MediaPipe FaceLandmarker ở chế độ `VIDEO`, 478 landmark, ma trận biến đổi, vòng `requestAnimationFrame` và EMA chống rung.
- `dist/ar/filterRenderer.js`: vẽ filter Canvas lên `#arCanvas` theo `chooseFilter()` (5 biểu cảm × nam/nữ = 10 filter, chủ đề chỉ đổi không khí nền; `dist/ar/filters/expressionMatrix.js`), chuyển filter bằng hiệu ứng mờ dần. Filter này luôn bật và được ghép vào ảnh chụp; chỉ nhường chỗ khi DeepAR đang vẽ filter mặt ở chủ đề Ngầu.
- `dist/ar/expressionFusion.js`: hợp nhất xác suất EfficientNet và blendshape theo trọng số 0,65 / 0,35, EMA; nhãn mới phải cao hơn nhãn hiện tại ít nhất 0,04 và giữ 140 ms mới được đổi.
- `dist/ar/filterPolicy.js`: ánh xạ thuần từ chủ đề + biểu cảm + giới tính đáng tin cậy sang cấu hình filter; giới tính chỉ dùng khi xác suất từ 0,75.
- `dist/ar/deepARAdapter.js`: bảng filter DeepAR của chủ đề Ngầu theo 5 biểu cảm × nam/nữ/trung tính, thay nền phía sau. Mục "Filter nâng cao" và "Nền phía sau người" chỉ hiện khi đã lưu License Key; nhập key tại `http://localhost:4173/#setup`.
- `dist/ar/ambientEffects.js`: hiệu ứng Canvas riêng cho từng chủ đề, giảm một nửa số hạt khi bật Reduce Motion.
- `dist/frameLayouts.js`: khung thành phẩm 1 ảnh lớn + 2 ảnh nhỏ, ảnh đặt kiểu `contain`.
- `dist/ar/filters/`: các module có chung giao diện `{ id, name, tags, draw(ctx, face, time) }`.
- `dist/vendor/mediapipe/` và `dist/assets/models/`: runtime WASM và model được phục vụ trên localhost, không cần CDN lúc chạy.

Nguồn biểu cảm chính là EfficientNet qua `/api/analyze`; blendshape MediaPipe là tín hiệu phụ chạy mỗi khung hình để giảm trễ cho biểu cảm rõ. Model chỉ có 5 biểu cảm (`happy`, `surprise`, `neutral`, `sad`, `angry`); mỗi biểu cảm có một filter cho nam và một cho nữ. Giới tính dưới 0,75 thì chỉ hiện viền màu theo biểu cảm. Nếu không thấy mặt hoặc đầu vào bị `LARGE_POSE`, `TOO_FAR`, `LOW_LIGHT`, `BLURRY`, canvas AR được ẩn ngay.

`sad` và `angry` là hai lớp yếu nhất của model và chưa được đo trên webcam. Vì vậy hai nhãn này chỉ được dùng khi model và blendshape cùng nghiêng về một phía; hiệu ứng tương ứng luôn nhẹ, vui hoặc động viên và không hiển thị nhãn tiêu cực lên người dùng.

## Kiểm tra policy

```powershell
cd sites\ai-smart-photo-booth
npm test
```

Lệnh trên chạy test `filterPolicy`, `expressionFusion`, `frameLayouts`, `deepARAdapter` và `ambientEffects` bằng Node test runner.

Ảnh camera và ảnh chụp chỉ nằm trong RAM của trình duyệt / tiến trình Python, không được ghi ra đĩa bởi server.
