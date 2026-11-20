import { FaceLandmarker, FilesetResolver } from "../vendor/mediapipe/vision_bundle.mjs";

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export class FaceTracker {
  constructor(video, canvas, { ema = 0.42, maxFps = 30 } = {}) {
    this.video = video;
    this.canvas = canvas;
    this.ema = ema;
    this.minFrameMs = 1000 / maxFps;
    this.landmarker = null;
    this.smoothed = null;
    this.lastVideoTime = -1;
    this.lastDetectAt = 0;
    this.lastFace = null;
    this.running = false;
    this.frameId = null;
  }

  async init() {
    if (this.landmarker) return;
    const vision = await FilesetResolver.forVisionTasks("/vendor/mediapipe/wasm");
    this.landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/assets/models/face_landmarker.task",
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true
    });
  }

  resize() {
    const width = this.video.videoWidth || 1280;
    const height = this.video.videoHeight || 720;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.smoothed = null;
    }
  }

  smoothLandmarks(raw) {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const next = raw.map(point => ({
      x: (1 - point.x) * width,
      y: point.y * height,
      z: point.z * width
    }));
    if (!this.smoothed || this.smoothed.length !== next.length) {
      this.smoothed = next;
      return next;
    }
    for (let index = 0; index < next.length; index += 1) {
      const previous = this.smoothed[index];
      previous.x += (next[index].x - previous.x) * this.ema;
      previous.y += (next[index].y - previous.y) * this.ema;
      previous.z += (next[index].z - previous.z) * this.ema;
    }
    return this.smoothed;
  }

  makeFace(result) {
    const raw = result.faceLandmarks?.[0];
    if (!raw?.length) {
      this.smoothed = null;
      return null;
    }
    const landmarks = this.smoothLandmarks(raw);
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const nose = landmarks[1];
    const chin = landmarks[152];
    const eyeMid = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };
    const eyeDistance = Math.max(distance(leftEye, rightEye), 1);
    // Tọa độ x đã được lật gương; dùng vector từ landmark 263 -> 33 để góc 0
    // vẫn là đầu thẳng. Chiều ngược lại tạo góc gần PI và lật sticker 180°.
    const roll = Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x);
    const yaw = clamp((nose.x - eyeMid.x) / eyeDistance * 1.8, -1, 1);
    const expectedNoseY = eyeMid.y + eyeDistance * 0.47;
    const pitch = clamp((nose.y - expectedNoseY) / eyeDistance * 1.5, -1, 1);
    const matrix = result.facialTransformationMatrixes?.[0]?.data || null;
    const categories = result.faceBlendshapes?.[0]?.categories || [];
    const rawBlendshapes = Object.fromEntries(categories.map(category => [category.categoryName, category.score]));
    return {
      landmarks,
      matrix,
      blendshapes: rawBlendshapes,
      roll,
      yaw,
      pitch,
      eyeDistance,
      center: { x: nose.x, y: (nose.y + chin.y) / 2 },
      width: this.canvas.width,
      height: this.canvas.height
    };
  }

  start(onFrame) {
    this.running = true;
    const loop = time => {
      if (!this.running) return;
      this.resize();
      const canDetect = this.landmarker && this.video.readyState >= 2 &&
        this.video.currentTime !== this.lastVideoTime && time - this.lastDetectAt >= this.minFrameMs;
      if (canDetect) {
        this.lastVideoTime = this.video.currentTime;
        this.lastDetectAt = time;
        try {
          this.lastFace = this.makeFace(this.landmarker.detectForVideo(this.video, time));
        } catch (error) {
          console.warn("FaceLandmarker frame skipped", error);
        }
      }
      onFrame(this.lastFace, time);
      this.frameId = requestAnimationFrame(loop);
    };
    this.frameId = requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
    if (this.frameId) cancelAnimationFrame(this.frameId);
    this.frameId = null;
    this.lastFace = null;
    this.smoothed = null;
    this.canvas.getContext("2d").clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
