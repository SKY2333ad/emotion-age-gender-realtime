export const FILTER_EXPRESSIONS = ["happy", "surprise", "neutral", "sad", "angry"];
export const FILTER_THEMES = ["cool", "birthday", "horror", "christmas"];
export const BLOCKING_STATUSES = new Set(["LARGE_POSE", "TOO_FAR", "LOW_LIGHT", "BLURRY"]);

export function isUsableProfile(profile) {
  return Boolean(profile?.detected && FILTER_EXPRESSIONS.includes(profile.expression) && !BLOCKING_STATUSES.has(profile.status));
}

export function chooseFilter(profile) {
  if (!isUsableProfile(profile)) {
    return { filterIds: [], primaryId: null, hidden: true, variant: "neutral", reason: profile?.message || "Chưa có đầu vào phù hợp" };
  }
  const theme = FILTER_THEMES.includes(profile.themeId) ? profile.themeId : "cool";
  const genderReliable = profile.genderProbability >= 0.75 && ["male", "female"].includes(profile.gender);
  return {
    filterIds: ["expression-matrix"],
    primaryId: "expression-matrix",
    hidden: false,
    theme,
    expression: profile.expression,
    gender: genderReliable ? profile.gender : null,
    variant: genderReliable ? (profile.gender === "male" ? "cool" : "warm") : "neutral",
    reason: `${theme} • ${profile.expression}`
  };
}
