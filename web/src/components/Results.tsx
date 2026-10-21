import { AnimatePresence, m } from "motion/react";
import { formatMs, pct } from '../lib/image';
import type { Box, Timing } from '../lib/types';

const ease = [0.16, 1, 0.3, 1] as const;

function TimingBar({ timing }: { timing: Timing }) {
  const parts = [
    { key: 'preprocess', label: 'Prepare', value: timing.preprocess, tone: 'bg-faint' },
    { key: 'inference', label: 'Model', value: timing.inference, tone: 'bg-accent' },
    { key: 'postprocess', label: 'Decode + NMS', value: timing.postprocess, tone: 'bg-muted' },
  ];
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-faint">Latency</span>
        <span className="font-mono text-sm tabular-nums">{formatMs(timing.total)}</span>
      </div>
      <div className="mt-2 flex h-1.5 overflow-hidden rounded-full bg-surface-2">
        {parts.map((p) => (
          <m.span
            key={p.key}
            className={`h-full ${p.tone}`}
            initial={{ width: 0 }}
            animate={{ width: `${(p.value / timing.total) * 100}%` }}
            transition={{ duration: 0.6, ease }}
          />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        {parts.map((p) => (
          <li key={p.key} className="flex items-center gap-1.5">
            <span className={`size-1.5 rounded-full ${p.tone}`} />
            {p.label} <span className="font-mono tabular-nums text-faint">{formatMs(p.value)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Results({
  boxes,
  crops,
  timing,
  onHover,
  live = false,
}: {
  boxes: Box[];
  crops: string[];
  timing: Timing | null;
  onHover?: (i: number | null) => void;
  live?: boolean;
}) {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <span className="font-mono text-[11px] uppercase tracking-wider text-faint">Detected</span>
        <p className="mt-1 text-3xl font-semibold tracking-tight tabular-nums">
          <AnimatePresence mode="popLayout" initial={false}>
            <m.span
              key={boxes.length}
              className="inline-block"
              initial={live ? false : { y: 12, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -12, opacity: 0 }}
              transition={{ duration: 0.3, ease }}
            >
              {boxes.length}
            </m.span>
          </AnimatePresence>{' '}
          <span className="text-lg font-normal text-muted">{boxes.length === 1 ? 'plate' : 'plates'}</span>
        </p>
      </div>

      {timing && <TimingBar timing={timing} />}

      {!live && (
        <ul className="flex flex-col gap-2">
          <AnimatePresence initial={false}>
            {boxes.map((b, i) => (
              <m.li
                key={`${i}-${Math.round(b.x)}`}
                layout
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4, delay: 0.06 * i, ease }}
                onPointerEnter={() => onHover?.(i)}
                onPointerLeave={() => onHover?.(null)}
                className="flex items-center gap-3 rounded-xl border border-line bg-surface-2/60 p-2.5 transition-colors hover:border-line-strong"
              >
                {crops[i] ? (
                  <img src={crops[i]} alt={`Plate ${i + 1}`} className="h-10 w-20 shrink-0 rounded-md object-cover ring-1 ring-line" />
                ) : (
                  <span className="h-10 w-20 shrink-0 rounded-md bg-surface" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-sm font-medium">Plate {i + 1}</span>
                    <span className="font-mono text-xs tabular-nums text-muted">{pct(b.score)}</span>
                  </div>
                  <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-surface">
                    <m.div
                      className="h-full rounded-full bg-accent"
                      initial={{ width: 0 }}
                      animate={{ width: pct(b.score) }}
                      transition={{ duration: 0.7, delay: 0.1 + 0.06 * i, ease }}
                    />
                  </div>
                  <p className="mt-1 font-mono text-[11px] text-faint">
                    {Math.round(b.w)}×{Math.round(b.h)} px at {Math.round(b.x)},{Math.round(b.y)}
                  </p>
                </div>
              </m.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
