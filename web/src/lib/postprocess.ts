import type { Box } from './types';

/** How the source image was fitted into the square model input. */
export type Letterbox = { scale: number; padX: number; padY: number };

export function letterbox(width: number, height: number, size: number): Letterbox {
  const scale = Math.min(size / width, size / height);
  return { scale, padX: (size - width * scale) / 2, padY: (size - height * scale) / 2 };
}

/**
 * Decodes raw YOLO output [1, 4 + classes, anchors] (single class here) into boxes
 * in source-image pixels, keeping those above `threshold`.
 */
export function decode(output: Float32Array, anchors: number, threshold: number, lb: Letterbox, width: number, height: number): Box[] {
  const boxes: Box[] = [];
  for (let i = 0; i < anchors; i++) {
    const score = output[4 * anchors + i];
    if (score < threshold) continue;
    const cx = output[i], cy = output[anchors + i], w = output[2 * anchors + i], h = output[3 * anchors + i];
    const x0 = Math.max(0, (cx - w / 2 - lb.padX) / lb.scale);
    const y0 = Math.max(0, (cy - h / 2 - lb.padY) / lb.scale);
    const x1 = Math.min(width, (cx + w / 2 - lb.padX) / lb.scale);
    const y1 = Math.min(height, (cy + h / 2 - lb.padY) / lb.scale);
    if (x1 - x0 < 2 || y1 - y0 < 2) continue;
    boxes.push({ x: x0, y: y0, w: x1 - x0, h: y1 - y0, score });
  }
  return boxes;
}

function iou(a: Box, b: Box) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

/** Greedy non-maximum suppression: keep the best box, drop others overlapping it by more than `threshold`. */
export function nms(boxes: Box[], threshold: number, limit = 50): Box[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: Box[] = [];
  for (const box of sorted) {
    if (kept.every((k) => iou(k, box) <= threshold)) kept.push(box);
    if (kept.length === limit) break;
  }
  return kept;
}
