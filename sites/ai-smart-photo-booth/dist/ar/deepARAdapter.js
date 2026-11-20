const SDK_URL = "https://cdn.jsdelivr.net/npm/deepar@5.6.22/js/deepar.esm.js";
const CDN_EFFECT_ROOT = "https://cdn.jsdelivr.net/npm/deepar@5.6.22/effects";
const LOCAL_EFFECT_ROOT = "/assets/deepar/cool";
const FACE_SLOT = "faceFilter";

const effect = (id, name, url) => Object.freeze({ id, name, url });

export function absoluteEffectUrl(url, baseUrl = globalThis.location?.href) {
  if (/^https?:\/\//i.test(url) || !baseUrl) return url;
  return new URL(url, baseUrl).href;
}

export const DEEPAR_COOL_EFFECTS = Object.freeze({
  happy: Object.freeze({
    male: effect("aviators", "Kính Aviators", `${CDN_EFFECT_ROOT}/aviators`),
    female: effect("pixel-hearts", "Tim pixel", `${LOCAL_EFFECT_ROOT}/8bitHearts.deepar`),
    neutral: effect("aviators", "Kính Aviators", `${CDN_EFFECT_ROOT}/aviators`)
  }),
  surprise: Object.freeze({
    male: effect("emotion-meter", "Emotion Meter", `${LOCAL_EFFECT_ROOT}/Emotion_Meter.deepar`),
    female: effect("emotion-exaggerator", "Emotions Exaggerator", `${LOCAL_EFFECT_ROOT}/Emotions_Exaggerator.deepar`),
    neutral: effect("emotion-meter", "Emotion Meter", `${LOCAL_EFFECT_ROOT}/Emotion_Meter.deepar`)
  }),
  neutral: Object.freeze({
    male: effect("viking-helmet", "Mũ Viking", `${LOCAL_EFFECT_ROOT}/viking_helmet.deepar`),
    female: effect("humanoid", "Humanoid", `${LOCAL_EFFECT_ROOT}/Humanoid.deepar`),
    neutral: effect("aviators", "Kính Aviators", `${CDN_EFFECT_ROOT}/aviators`)
  }),
  sad: Object.freeze({
    male: effect("hope", "Hope", `${LOCAL_EFFECT_ROOT}/Hope.deepar`),
    female: effect("galaxy", "Galaxy Background", `${CDN_EFFECT_ROOT}/galaxy_background`),
    neutral: effect("hope", "Hope", `${LOCAL_EFFECT_ROOT}/Hope.deepar`)
  }),
  angry: Object.freeze({
    male: effect("fire", "Fire Effect", `${LOCAL_EFFECT_ROOT}/Fire_Effect.deepar`),
    female: effect("neon-devil-horns", "Neon Devil Horns", `${LOCAL_EFFECT_ROOT}/Neon_Devil_Horns.deepar`),
    neutral: effect("fire", "Fire Effect", `${LOCAL_EFFECT_ROOT}/Fire_Effect.deepar`)
  })
});

export function deepArGenderVariant(gender, probability = 0) {
  if (Number(probability) < 0.75) return "neutral";
  return gender === "male" || gender === "female" ? gender : "neutral";
}

export function resolveCoolDeepAREffect(profile = {}) {
  if (profile.themeId !== "cool") return null;
  const expression = DEEPAR_COOL_EFFECTS[profile.expression] ? profile.expression : "neutral";
  const variant = deepArGenderVariant(profile.gender, profile.genderProbability);
  const selected = DEEPAR_COOL_EFFECTS[expression][variant];
  return { ...selected, expression, variant, key: `${expression}:${variant}:${selected.id}` };
}

export class DeepARAdapter {
  constructor(video, canvas) {
    this.video = video;
    this.canvas = canvas;
    this.instance = null;
    this.initializing = null;
    this.enabled = false;
    this.activeEffect = null;
    this.requestedEffect = null;
    this.switching = null;
    this.activeBackground = null;
  }

  get ready() { return Boolean(this.instance); }

  async initialize(licenseKey) {
    if (this.instance) return this.instance;
    if (this.initializing) return this.initializing;
    if (!licenseKey?.trim()) throw new Error("Thiếu DeepAR SDK License Key.");
    this.initializing = (async () => {
      const deepar = await import(SDK_URL);
      this.canvas.width = this.video.videoWidth || 1280;
      this.canvas.height = this.video.videoHeight || 720;
      const instance = await deepar.initialize({
        licenseKey: licenseKey.trim(), canvas: this.canvas,
        additionalOptions: { cameraConfig: { disableDefaultCamera: true } }
      });
      instance.setVideoElement(this.video, true);
      this.instance = instance;
      return instance;
    })();
    try { return await this.initializing; } finally { this.initializing = null; }
  }

  async enable(licenseKey) {
    const instance = await this.initialize(licenseKey);
    instance.setVideoElement(this.video, true);
    instance.setPaused?.(false);
    this.canvas.hidden = false;
    this.enabled = true;
  }

  async setProfile(profile) {
    const selected = resolveCoolDeepAREffect(profile);
    if (!this.enabled || !this.instance || !selected) return false;
    this.requestedEffect = selected;
    if (this.activeEffect?.key === selected.key || this.switching) return false;
    this.switching = (async () => {
      let changed = false;
      while (this.enabled && this.requestedEffect && this.requestedEffect.key !== this.activeEffect?.key) {
        const next = this.requestedEffect;
        await this.instance.switchEffect(absoluteEffectUrl(next.url), { slot:FACE_SLOT });
        this.activeEffect = next;
        changed = true;
      }
      return changed;
    })();
    try { return await this.switching; } finally { this.switching = null; }
  }

  async clearFaceEffect() {
    if (!this.instance) return;
    await this.instance.clearEffect?.(FACE_SLOT);
    this.activeEffect = null;
    this.requestedEffect = null;
  }

  async setBackground(imageUrl) {
    if (!this.enabled || !this.instance) throw new Error("DeepAR chưa sẵn sàng.");
    const url = absoluteEffectUrl(imageUrl);
    await this.instance.backgroundReplacement(true, url);
    this.activeBackground = url;
  }

  async clearBackground() {
    if (!this.instance || !this.activeBackground) return;
    await this.instance.backgroundReplacement(false, this.activeBackground);
    this.activeBackground = null;
  }

  pause() { this.enabled = false; this.canvas.hidden = true; this.instance?.setPaused?.(true); }
  async takeScreenshot() { return this.enabled && this.instance ? this.instance.takeScreenshot() : null; }
  shutdown() {
    this.enabled = false; this.canvas.hidden = true; this.instance?.shutdown?.();
    this.instance = null; this.activeEffect = null; this.requestedEffect = null; this.switching = null; this.activeBackground = null;
  }
}

export const DEEPAR_LICENSE_STORAGE_KEY = "photoBooth.deepARLicenseKey";
