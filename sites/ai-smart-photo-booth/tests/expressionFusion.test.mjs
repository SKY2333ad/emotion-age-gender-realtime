import test from "node:test";
import assert from "node:assert/strict";
import { createFusionState, updateExpressionFusion } from "../dist/ar/expressionFusion.js";

const probs = values => ({ happy:0.02, surprise:0.02, neutral:0.02, sad:0.02, angry:0.02, ...values });
const settle = (input, label = "neutral") => {
  let output = { state:createFusionState(label) };
  for (let t = 0; t <= 300; t += 50) output = updateExpressionFusion(output.state, input, t);
  return output.result;
};

test("only the five project expressions are scored", () => {
  const output = updateExpressionFusion(createFusionState("neutral"), { probs:probs({happy:.9}), modelToken:1 }, 0);
  assert.deepEqual(Object.keys(output.result.scores).sort(), ["angry", "happy", "neutral", "sad", "surprise"]);
});

test("sad and angry follow the model without needing blendshapes", () => {
  assert.equal(settle({ probs:probs({sad:.7}), modelToken:1, blendshapes:{} }).label, "sad");
  assert.equal(settle({ probs:probs({angry:.6}), modelToken:1, blendshapes:{} }).label, "angry");
});

test("low model confidence falls back to neutral", () => {
  assert.equal(settle({ probs:{ happy:.3, surprise:.25, neutral:.15, sad:.15, angry:.15 }, modelToken:1 }, "happy").label, "neutral");
});

test("new label switches after the 140ms hold", () => {
  const input = { probs:probs({happy:.9}), modelToken:1 };
  const first = updateExpressionFusion(createFusionState("neutral"), input, 0);
  assert.equal(first.result.label, "neutral");
  const early = updateExpressionFusion(first.state, input, 100);
  assert.equal(early.result.label, "neutral");
  assert.equal(updateExpressionFusion(early.state, input, 150).result.label, "happy");
});
