import { FILTER_MAP } from "./filters/index.js";

export class FilterRenderer {
  constructor(canvas, { transitionMs = 520 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true, desynchronized: true });
    this.transitionMs = transitionMs;
    this.current = { filterIds: [], variant: "neutral" };
    this.previous = null;
    this.transitionAt = 0;
  }

  setSelection(selection, now = performance.now()) {
    const next = selection || { filterIds: [], variant: "neutral" };
    const currentKey = JSON.stringify(this.current);
    const nextKey = JSON.stringify(next);
    if (currentKey === nextKey) return false;
    this.previous = this.current;
    this.current = { ...next, filterIds: [...next.filterIds], variant: next.variant || "neutral" };
    this.transitionAt = now;
    return true;
  }

  drawSelection(selection, face, time, alpha) {
    if (!face || alpha <= 0) return;
    this.ctx.save();
    this.ctx.globalAlpha = alpha;
    const renderFace = { ...face, ...selection, variant: selection.variant };
    for (const id of selection.filterIds) FILTER_MAP.get(id)?.draw(this.ctx, renderFace, time);
    this.ctx.restore();
  }

  draw(face, time, visible = true) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!visible || !face) return;
    const progress = Math.min(1, Math.max(0, (time - this.transitionAt) / this.transitionMs));
    if (this.previous && progress < 1) this.drawSelection(this.previous, face, time, 1 - progress);
    this.drawSelection(this.current, face, time, this.previous ? progress : 1);
    if (progress >= 1) this.previous = null;
  }
}
