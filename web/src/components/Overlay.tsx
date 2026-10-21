import { AnimatePresence, m } from "motion/react";
import type { Box } from '../lib/types';

const ease = [0.16, 1, 0.3, 1] as const;

/** Corner brackets: an L at each corner, sized to the box. */
function brackets(b: Box) {
  const l = Math.min(b.w, b.h) * 0.35;
  const { x, y, w, h } = b;
  return [
    `M${x} ${y + l}V${y}H${x + l}`,
    `M${x + w - l} ${y}H${x + w}V${y + l}`,
    `M${x + w} ${y + h - l}V${y + h}H${x + w - l}`,
    `M${x + l} ${y + h}H${x}V${y + h - l}`,
  ].join('');
}

type Rect = { x: number; y: number; w: number; h: number };
const overlaps = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/**
 * Picks a spot for each confidence tag: above its box, else below, else inside the top
 * edge, taking the first that covers no other plate or earlier tag and stays in frame.
 */
function placeTags(boxes: Box[], w: number, h: number, gap: number, width: number, height: number): Rect[] {
  const placed: Rect[] = [];
  return boxes.map((b, i) => {
    const x = Math.min(Math.max(0, b.x), width - w);
    const candidates = [
      { x, y: b.y - h - gap },
      { x, y: b.y + b.h + gap },
      { x: b.x + gap, y: b.y + gap },
    ];
    const free = candidates.find((c) => {
      const r = { ...c, w, h };
      return c.y >= 0 && c.y + h <= height && boxes.every((o, j) => j === i || !overlaps(r, o)) && placed.every((p) => !overlaps(r, p));
    });
    const tag = { ...(free ?? { x, y: Math.max(0, b.y - h - gap) }), w, h };
    placed.push(tag);
    return tag;
  });
}

/**
 * SVG drawn over the image in source-pixel coordinates, so boxes line up at any
 * display size. `live` skips the entrance animation (webcam frames arrive too fast for it).
 */
export function Overlay({
  width,
  height,
  boxes,
  highlighted,
  scanning,
  live = false,
  showScore = true,
}: {
  width: number;
  height: number;
  boxes: Box[];
  highlighted?: number | null;
  scanning?: boolean;
  live?: boolean;
  /** Off for ground-truth boxes, which have no confidence. */
  showScore?: boolean;
}) {
  const label = Math.max(12, Math.round(Math.min(width, height) / 38));
  const tags = placeTags(boxes, label * (showScore ? 5.4 : 3.4), label * 1.45, label * 0.25, width, height);

  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      {scanning && (
        <m.g initial={{ y: 0 }} animate={{ y: height }} transition={{ duration: 1.1, ease: 'easeInOut', repeat: Infinity, repeatType: 'reverse' }}>
          <rect x={0} y={-height * 0.08} width={width} height={height * 0.08} fill="url(#scan)" />
          <line x1={0} x2={width} y1={0} y2={0} stroke="var(--accent)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        </m.g>
      )}
      <defs>
        <linearGradient id="scan" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0" />
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0.28" />
        </linearGradient>
      </defs>

      <AnimatePresence>
        {boxes.map((b, i) => {
          const on = highlighted == null || highlighted === i;
          const delay = live ? 0 : 0.08 * i;
          return (
            <m.g
              key={live ? i : `${i}-${Math.round(b.x)}-${Math.round(b.y)}`}
              initial={live ? false : { opacity: 0 }}
              animate={{ opacity: on ? 1 : 0.3 }}
              exit={{ opacity: 0, transition: { duration: 0.15 } }}
              transition={{ duration: 0.2 }}
            >
              <m.rect
                x={b.x}
                y={b.y}
                width={b.w}
                height={b.h}
                rx={Math.min(b.w, b.h) * 0.06}
                fill="var(--accent-soft)"
                stroke="var(--accent)"
                strokeOpacity={0.55}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
                initial={live ? false : { pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.6, delay, ease }}
              />
              <m.path
                d={brackets(b)}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={3}
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
                initial={live ? false : { opacity: 0, scale: 1.15 }}
                animate={{ opacity: 1, scale: 1 }}
                style={{ transformOrigin: `${b.x + b.w / 2}px ${b.y + b.h / 2}px`, transformBox: 'view-box' }}
                transition={{ duration: 0.45, delay: delay + 0.25, ease }}
              />
              <m.g
                initial={live ? false : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: delay + 0.4, ease }}
              >
                <rect x={tags[i].x} y={tags[i].y} width={tags[i].w} height={tags[i].h} rx={label * 0.3} fill="var(--accent)" />
                <text
                  x={tags[i].x + label * 0.45}
                  y={tags[i].y + label * 1.05}
                  fontSize={label}
                  fontFamily="JetBrains Mono Variable, monospace"
                  fontWeight={600}
                  fill="var(--on-accent)"
                >
                  {showScore ? `PLATE ${Math.round(b.score * 100)}%` : 'PLATE'}
                </text>
              </m.g>
            </m.g>
          );
        })}
      </AnimatePresence>
    </svg>
  );
}
