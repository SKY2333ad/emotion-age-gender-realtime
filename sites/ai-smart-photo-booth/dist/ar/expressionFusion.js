export const EXPRESSION_LABELS = ["happy", "surprise", "neutral", "sad", "angry"];

export const CONFIG = Object.freeze({
  modelWeight: 0.65,
  blendshapeWeight: 0.35,
  modelEmaAlpha: 0.72,
  blendshapeEmaAlpha: 0.48,
  switchMargin: 0.04,
  holdMs: 140,
  lowModelConfidence: 0.45,
  clearBlendshape: 0.35,
  negativeModelMin: 0.24,
  negativeBlendshapeMin: 0.2,
  thresholds: Object.freeze({
    smile: 0.5,
    jawOpen: 0.3,
    browInnerUp: 0.35,
    noseSneer: 0.35,
    browDown: 0.5,
    mouthPress: 0.25,
    mouthFrown: 0.3,
    lowSmile: 0.15
  })
});

const clamp01 = value => Math.min(1, Math.max(0, Number(value) || 0));
const mean = (...values) => values.reduce((total, value) => total + clamp01(value), 0) / values.length;
const ramp = (value, threshold, high = 1) => clamp01((value - threshold) / Math.max(high - threshold, 1e-6));

export function blendshapeExpressionScores(shapes = {}, config = CONFIG) {
  const threshold = config.thresholds;
  const smile = mean(shapes.mouthSmileLeft, shapes.mouthSmileRight);
  const frown = mean(shapes.mouthFrownLeft, shapes.mouthFrownRight);
  const sneer = mean(shapes.noseSneerLeft, shapes.noseSneerRight);
  const browDown = mean(shapes.browDownLeft, shapes.browDownRight);
  const mouthPress = mean(shapes.mouthPressLeft, shapes.mouthPressRight);
  const happy = ramp(smile, threshold.smile, 0.9);
  const surprise = Math.sqrt(ramp(shapes.jawOpen, threshold.jawOpen, 0.75) * ramp(shapes.browInnerUp, threshold.browInnerUp, 0.8)) * (1 - smile * 0.72);
  const angry = Math.sqrt(ramp(browDown, threshold.browDown, 0.9) * Math.max(ramp(mouthPress, threshold.mouthPress, 0.7), ramp(sneer, threshold.noseSneer, 0.75))) * (1 - smile * 0.7);
  const sad = ramp(frown, threshold.mouthFrown, 0.72) * clamp01((threshold.lowSmile - smile) / threshold.lowSmile);
  const strongest = Math.max(happy, surprise, angry, sad);
  return { happy, surprise, angry, sad, neutral: clamp01(1 - strongest * 1.25) };
}

function ema(previous, values, alpha) {
  if (!previous) return Object.fromEntries(EXPRESSION_LABELS.map(label => [label, clamp01(values[label])]));
  return Object.fromEntries(EXPRESSION_LABELS.map(label => [label, previous[label] + alpha * (clamp01(values[label]) - previous[label])]));
}

function topEntry(scores, allowed = EXPRESSION_LABELS) {
  return allowed.reduce((best, label) => scores[label] > best.score ? { label, score: scores[label] } : best, { label: "neutral", score: -1 });
}

export function createFusionState(initialLabel = "neutral") {
  return { modelEma: null, blendEma: null, modelToken: null, currentLabel: initialLabel, candidateLabel: null, candidateSince: null };
}

export function updateExpressionFusion(state, input, nowMs, config = CONFIG) {
  const next = { ...state };
  const rawModel = Object.fromEntries(EXPRESSION_LABELS.map(label => [label, clamp01(input.probs?.[label])]));
  if (input.modelToken !== state.modelToken) {
    next.modelEma = ema(state.modelEma, rawModel, config.modelEmaAlpha);
    next.modelToken = input.modelToken;
  }
  next.modelEma ||= rawModel;
  const rawBlend = blendshapeExpressionScores(input.blendshapes, config);
  next.blendEma = ema(state.blendEma, rawBlend, config.blendshapeEmaAlpha);

  const fused = Object.fromEntries(EXPRESSION_LABELS.map(label => [
    label,
    config.modelWeight * next.modelEma[label] + config.blendshapeWeight * next.blendEma[label]
  ]));

  const modelTop = topEntry(next.modelEma);
  if (next.blendEma.happy >= 0.5 && ["sad", "angry"].includes(modelTop.label)) fused.happy = Math.max(fused.happy, 0.68);
  if (next.blendEma.surprise >= 0.5 && modelTop.label !== "happy") fused.surprise = Math.max(fused.surprise, 0.64);

  for (const label of ["sad", "angry"]) {
    if (next.modelEma[label] < config.negativeModelMin || next.blendEma[label] < config.negativeBlendshapeMin) fused[label] = Math.min(fused[label], 0.18);
  }

  const maxModel = Math.max(...Object.values(next.modelEma));
  const maxClearBlend = Math.max(next.blendEma.happy, next.blendEma.surprise, next.blendEma.angry);
  if (maxModel < config.lowModelConfidence && maxClearBlend < config.clearBlendshape) fused.neutral = Math.max(fused.neutral, 0.7);

  const target = topEntry(fused);
  const currentScore = fused[next.currentLabel] || 0;
  const enoughMargin = target.label !== next.currentLabel && target.score - currentScore >= config.switchMargin;
  if (!enoughMargin) {
    next.candidateLabel = null;
    next.candidateSince = null;
  } else if (next.candidateLabel !== target.label) {
    next.candidateLabel = target.label;
    next.candidateSince = nowMs;
  } else if (nowMs - next.candidateSince >= config.holdMs) {
    next.currentLabel = target.label;
    next.candidateLabel = null;
    next.candidateSince = null;
  }
  return { state: next, result: debugResult(next.currentLabel, next, fused, rawBlend, "fused") };
}

function debugResult(label, state, fused, rawBlend, reason) {
  return {
    label,
    score: fused[label] || 0,
    modelProbability: state.modelEma?.[label] || 0,
    blendshapeScore: state.blendEma?.[label] || 0,
    rawBlendshapeScore: rawBlend[label] || 0,
    scores: fused,
    reason
  };
}
