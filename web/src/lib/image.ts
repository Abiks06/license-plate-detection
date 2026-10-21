import type { Box } from './types';

/** Grows a box by `pad` of its size on each side, clamped to the image. */
function padded(b: Box, width: number, height: number, pad: number) {
  const px = b.w * pad, py = b.h * pad;
  const x = Math.max(0, b.x - px), y = Math.max(0, b.y - py);
  return { x, y, w: Math.min(width, b.x + b.w + px) - x, h: Math.min(height, b.y + b.h + py) - y };
}

/** Small JPEG data URLs of each detected plate, for the results list. */
export function cropPlates(image: CanvasImageSource, width: number, height: number, boxes: Box[], maxW = 220) {
  return boxes.map((b) => {
    const r = padded(b, width, height, 0.15);
    const scale = Math.min(1, maxW / r.w);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(r.w * scale));
    canvas.height = Math.max(1, Math.round(r.h * scale));
    canvas.getContext('2d')!.drawImage(image, r.x, r.y, r.w, r.h, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  });
}

/**
 * Draws the image with every plate pixelated beyond recognition. Pixelation
 * (rather than blur) can't be reversed by sharpening.
 */
export function redact(image: CanvasImageSource, width: number, height: number, boxes: Box[]) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(image, 0, 0, width, height);

  const tiny = document.createElement('canvas');
  const t = tiny.getContext('2d')!;
  for (const b of boxes) {
    const r = padded(b, width, height, 0.12);
    // ~6 blocks across the plate: enough to read as "redacted", far too few to read characters.
    const blocks = 6;
    tiny.width = blocks;
    tiny.height = Math.max(2, Math.round((blocks * r.h) / r.w));
    t.imageSmoothingEnabled = true;
    t.drawImage(canvas, r.x, r.y, r.w, r.h, 0, 0, tiny.width, tiny.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tiny, 0, 0, tiny.width, tiny.height, r.x, r.y, r.w, r.h);
  }
  return canvas;
}

export function download(canvas: HTMLCanvasElement, filename: string) {
  canvas.toBlob(
    (blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = Object.assign(document.createElement('a'), { href: url, download: filename });
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
    'image/jpeg',
    0.92,
  );
}

export const formatBytes = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.round(n / 1e3)} KB`);
export const formatMs = (n: number) => (n >= 100 ? `${Math.round(n)} ms` : `${n.toFixed(1)} ms`);
export const pct = (n: number) => `${Math.round(n * 100)}%`;
