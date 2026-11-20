import test from "node:test";
import assert from "node:assert/strict";
import { chooseFilter, isUsableProfile, manualProfile } from "../dist/ar/filterPolicy.js";

const profile = overrides => ({ detected:true, expression:"happy", themeId:"cool", gender:"female", genderProbability:.9, status:"OK", ...overrides });

test("maps a valid profile to the theme x expression matrix", () => {
  const result=chooseFilter(profile({themeId:"birthday",expression:"surprise"}));
  assert.deepEqual(result.filterIds,["expression-matrix"]);
  assert.equal(result.theme,"birthday");
  assert.equal(result.expression,"surprise");
});

test("supports horror and Christmas themes", () => {
  assert.equal(chooseFilter(profile({themeId:"horror"})).theme,"horror");
  assert.equal(chooseFilter(profile({themeId:"christmas"})).theme,"christmas");
});

test("gender is ignored below 0.75 confidence", () => {
  const result=chooseFilter(profile({genderProbability:.74}));
  assert.equal(result.gender,null);
  assert.equal(result.variant,"neutral");
});

test("gender only changes the auxiliary style variant when reliable", () => {
  assert.equal(chooseFilter(profile({gender:"male",genderProbability:.8})).variant,"cool");
  assert.equal(chooseFilter(profile({gender:"female",genderProbability:.8})).variant,"warm");
});

test("blocking status hides filters", () => {
  const result=chooseFilter(profile({status:"LOW_LIGHT"}));
  assert.equal(result.hidden,true);
  assert.deepEqual(result.filterIds,[]);
});

test("removed expressions fear and disgust have no filter", () => {
  assert.equal(isUsableProfile(profile({expression:"fear"})),false);
  assert.equal(isUsableProfile(profile({expression:"disgust"})),false);
});

test("manual filter choice works without emotion recognition", () => {
  const face = { landmarks: [] };
  const selected = manualProfile("female:sad", face, "christmas");
  assert.equal(selected.ready, true);
  const filter = chooseFilter(selected);
  assert.equal(filter.expression, "sad");
  assert.equal(filter.gender, "female");
  assert.equal(filter.theme, "christmas");
  assert.equal(manualProfile("female:sad", null, "cool").ready, false);
  assert.equal(manualProfile("", face, "cool"), null);
  assert.equal(manualProfile("male:fear", face, "cool"), null);
});
