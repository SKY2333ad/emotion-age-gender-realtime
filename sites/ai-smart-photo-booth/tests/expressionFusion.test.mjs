import test from "node:test";
import assert from "node:assert/strict";
import { blendshapeExpressionScores, createFusionState, updateExpressionFusion } from "../dist/ar/expressionFusion.js";

const probs = values => ({ happy:0.02, surprise:0.02, neutral:0.02, sad:0.02, angry:0.02, ...values });

test("blendshape thresholds detect a clear smile", () => {
  const scores = blendshapeExpressionScores({ mouthSmileLeft:.82, mouthSmileRight:.78 });
  assert.ok(scores.happy > .5);
});

test("only the five project expressions are scored", () => {
  const output = updateExpressionFusion(createFusionState("neutral"), { probs:probs({happy:.9}), modelLabel:"happy", modelToken:1, blendshapes:{} }, 0);
  assert.deepEqual(Object.keys(output.result.scores).sort(), ["angry", "happy", "neutral", "sad", "surprise"]);
});

test("low model confidence and unclear blendshapes fall back to neutral", () => {
  const state = createFusionState("happy");
  const first = updateExpressionFusion(state, { probs:probs({happy:.2,neutral:.25}), modelLabel:"neutral", modelToken:1, blendshapes:{} }, 0);
  const second = updateExpressionFusion(first.state, { probs:probs({happy:.2,neutral:.25}), modelLabel:"neutral", modelToken:1, blendshapes:{} }, 260);
  assert.equal(second.result.label, "neutral");
});

test("new label switches after the faster 140ms hold", () => {
  const state = createFusionState("neutral");
  const input = { probs:probs({happy:.9}), modelLabel:"happy", modelToken:1, blendshapes:{mouthSmileLeft:.9,mouthSmileRight:.9} };
  const first = updateExpressionFusion(state, input, 0);
  assert.equal(first.result.label, "neutral");
  const early = updateExpressionFusion(first.state, input, 100);
  assert.equal(early.result.label, "neutral");
  const ready = updateExpressionFusion(early.state, input, 150);
  assert.equal(ready.result.label, "happy");
});

test("sad needs both model and blendshape support", () => {
  const input = { probs:probs({sad:.85}), modelLabel:"sad", modelToken:1, blendshapes:{mouthSmileLeft:.05,mouthSmileRight:.05,mouthFrownLeft:.05,mouthFrownRight:.05} };
  const output = updateExpressionFusion(createFusionState("neutral"), input, 0);
  assert.ok(output.result.scores.sad <= .18);
});

test("weights combine model and blendshape using configured 0.65/0.35 ratio", () => {
  const input = { probs:probs({happy:1}), modelLabel:"happy", modelToken:1, blendshapes:{mouthSmileLeft:.5,mouthSmileRight:.5} };
  const output = updateExpressionFusion(createFusionState("neutral"), input, 0);
  assert.ok(Math.abs(output.result.scores.happy-.65)<1e-9);
});
