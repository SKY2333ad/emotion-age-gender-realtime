export const EXPRESSION_LABELS = ["happy", "surprise", "neutral", "sad", "angry"];

// Chỉ dùng xác suất của model EfficientNet: blendshape MediaPipe gần như không bao giờ
// thấy buồn/giận, nên gộp với blendshape từng làm mất hẳn hai cảm xúc này.
export const CONFIG = Object.freeze({
  modelEmaAlpha: 0.72,
  switchMargin: 0.04,
  holdMs: 140,
  lowModelConfidence: 0.35
});

const clamp01 = value => Math.min(1, Math.max(0, Number(value) || 0));

function ema(previous, values, alpha) {
  if (!previous) return values;
  return Object.fromEntries(EXPRESSION_LABELS.map(label => [label, previous[label] + alpha * (values[label] - previous[label])]));
}

function topEntry(scores) {
  return EXPRESSION_LABELS.reduce((best, label) => scores[label] > best.score ? { label, score: scores[label] } : best, { label: "neutral", score: -1 });
}

export function createFusionState(initialLabel = "neutral") {
  return { modelEma: null, modelToken: null, currentLabel: initialLabel, candidateLabel: null, candidateSince: null };
}

export function updateExpressionFusion(state, input, nowMs, config = CONFIG) {
  const next = { ...state };
  const rawModel = Object.fromEntries(EXPRESSION_LABELS.map(label => [label, clamp01(input.probs?.[label])]));
  if (input.modelToken !== state.modelToken || !next.modelEma) {
    next.modelEma = ema(state.modelEma, rawModel, config.modelEmaAlpha);
    next.modelToken = input.modelToken;
  }
  const scores = { ...next.modelEma };
  // Model phân vân giữa nhiều cảm xúc thì coi là bình thản.
  if (topEntry(scores).score < config.lowModelConfidence) scores.neutral = Math.max(scores.neutral, config.lowModelConfidence);

  const target = topEntry(scores);
  const currentScore = scores[next.currentLabel] || 0;
  if (target.label === next.currentLabel || target.score - currentScore < config.switchMargin) {
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
  const label = next.currentLabel;
  return { state: next, result: { label, score: scores[label] || 0, scores } };
}
