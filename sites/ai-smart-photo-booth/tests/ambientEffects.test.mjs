import test from "node:test";
import assert from "node:assert/strict";
import { effectForTheme } from "../dist/ar/ambientEffects.js";

test("each Photo Booth theme has a lightweight ambient canvas effect", () => {
  for (const theme of ["cool", "birthday", "horror", "christmas"]) {
    const effect = effectForTheme(theme);
    assert.ok(effect.count >= 20);
    assert.ok(effect.colors.length >= 3);
  }
});

test("unknown themes use the neon effect safely", () => {
  assert.deepEqual(effectForTheme("unknown"), effectForTheme("cool"));
});

test("the three expressive themes have distinct visual identities", () => {
  assert.equal(effectForTheme("cool").style, "cyber-neon");
  assert.equal(effectForTheme("birthday").style, "party-pop");
  assert.equal(effectForTheme("horror").style, "haunted-fog");
});
