import test from "node:test";
import assert from "node:assert/strict";
import {
  DeepARAdapter, DEEPAR_COOL_EFFECTS, absoluteEffectUrl, deepArGenderVariant, resolveCoolDeepAREffect
} from "../dist/ar/deepARAdapter.js";

function readyAdapter() {
  const calls = [];
  const adapter = new DeepARAdapter({}, { hidden:false });
  adapter.enabled = true;
  adapter.activeEffect = resolveCoolDeepAREffect({ themeId:"cool", expression:"neutral" });
  adapter.requestedEffect = adapter.activeEffect;
  adapter.instance = { async switchEffect(url) { calls.push(url); } };
  return { adapter, calls };
}

test("has male and female Cool filters for all five expressions", () => {
  const expressions = ["happy", "surprise", "neutral", "sad", "angry"];
  for (const expression of expressions) {
    assert.ok(DEEPAR_COOL_EFFECTS[expression].male.url);
    assert.ok(DEEPAR_COOL_EFFECTS[expression].female.url);
    assert.notEqual(DEEPAR_COOL_EFFECTS[expression].male.url, DEEPAR_COOL_EFFECTS[expression].female.url);
  }
});

test("uses gender only from 75 percent confidence", () => {
  assert.equal(deepArGenderVariant("male", .74), "neutral");
  assert.equal(deepArGenderVariant("male", .75), "male");
  assert.equal(deepArGenderVariant("female", .91), "female");
  assert.equal(deepArGenderVariant(null, 1), "neutral");
});

test("converts local DeepAR assets to absolute URLs for the SDK worker", () => {
  assert.equal(
    absoluteEffectUrl("/assets/deepar/cool/Hope.deepar", "https://photo.example/capture"),
    "https://photo.example/assets/deepar/cool/Hope.deepar"
  );
});

test("resolves effects only for the Cool theme", () => {
  assert.equal(resolveCoolDeepAREffect({ themeId:"birthday", expression:"happy", gender:"female", genderProbability:.9 }), null);
  const selected = resolveCoolDeepAREffect({ themeId:"cool", expression:"happy", gender:"female", genderProbability:.9 });
  assert.equal(selected.id, "pixel-hearts");
  assert.equal(selected.variant, "female");
});

test("adapter switches when expression or reliable gender changes", async () => {
  const { adapter, calls } = readyAdapter();
  const femaleHappy = { themeId:"cool", expression:"happy", gender:"female", genderProbability:.9 };
  assert.equal(await adapter.setProfile(femaleHappy), true);
  assert.deepEqual(calls, [DEEPAR_COOL_EFFECTS.happy.female.url]);
  assert.equal(await adapter.setProfile(femaleHappy), false);
  assert.equal(calls.length, 1);
  assert.equal(await adapter.setProfile({ ...femaleHappy, gender:"male" }), true);
  assert.equal(calls.at(-1), DEEPAR_COOL_EFFECTS.happy.male.url);
});

test("adapter enables and clears a custom background behind the person", async () => {
  const calls = [];
  const adapter = new DeepARAdapter({}, { hidden:false });
  adapter.enabled = true;
  adapter.instance = { async backgroundReplacement(...args) { calls.push(args); } };
  await adapter.setBackground("/assets/backgrounds/cool-neon-city.png");
  assert.equal(calls[0][0], true);
  assert.match(calls[0][1], /assets\/backgrounds\/cool-neon-city\.png$/);
  await adapter.clearBackground();
  assert.equal(calls[1][0], false);
  assert.equal(adapter.activeBackground, null);
});

