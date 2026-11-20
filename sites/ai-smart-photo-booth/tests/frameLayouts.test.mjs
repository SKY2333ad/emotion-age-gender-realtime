import test from "node:test";
import assert from "node:assert/strict";
import { FRAME_LAYOUTS, getFrameLayout } from "../dist/frameLayouts.js";

test("each theme has its own three-photo frame", () => {
  assert.deepEqual(Object.keys(FRAME_LAYOUTS), ["cool", "birthday", "horror", "christmas"]);
  for (const layout of Object.values(FRAME_LAYOUTS)) {
    assert.equal(layout.photos.length, 3);
    assert.ok(layout.photos.every(photo => photo.w > 0 && photo.h > 0));
    assert.ok(layout.photos.every(photo => photo.fit === "contain"));
    assert.ok(layout.photos.every(photo => Math.abs(photo.w / photo.h - 16 / 9) < 0.01));
  }
  assert.equal(new Set(Object.values(FRAME_LAYOUTS).map(layout => JSON.stringify(layout.photos))).size, 4);
});

test("unknown themes fall back to Cool", () => {
  assert.equal(getFrameLayout("unknown").id, "cool");
});
