import test from "node:test";
import assert from "node:assert/strict";
import { GENDER_FILTERS, genderFilterFor } from "../dist/ar/filters/expressionMatrix.js";
import { FILTER_EXPRESSIONS } from "../dist/ar/filterPolicy.js";

test("each of the five expressions has one male and one female filter", () => {
  const names = [];
  for (const gender of ["male", "female"]) {
    assert.deepEqual(Object.keys(GENDER_FILTERS[gender]).sort(), [...FILTER_EXPRESSIONS].sort());
    for (const expression of FILTER_EXPRESSIONS) names.push(genderFilterFor(gender, expression).name);
  }
  assert.equal(new Set(names).size, 10);
});

test("uncertain gender gets no gender-specific filter", () => {
  assert.equal(genderFilterFor(null, "happy"), null);
});
