import { AnimatePresence, m } from "motion/react";
import { Check, Copy } from 'lucide-react';
import { useState } from 'react';
import { formatMs, pct } from '../lib/image';
import { charSlots, type PlateRead } from '../lib/plateText';
import type { Box, Timing } from '../lib/types';

const ease = [0.16, 1, 0.3, 1] as const;
/** Characters the reader is less sure of than this are marked for a second look. */
const DOUBT = 0.6;

function TimingBar({ timing }: { timing: Timing }) {
  const parts = [
    { key: 'preprocess', label: 'Prepare', value: timing.preprocess, tone: 'bg-faint' },
    { key: 'inference', label: 'Detect', value: timing.inference, tone: 'bg-accent' },
    { key: 'postprocess', label: 'Decode + NMS', value: timing.postprocess, tone: 'bg-muted' },
    { key: 'read', label: 'Read', value: timing.read, tone: 'bg-text/70' },
  ].filter((p) => p.key !== 'read' || p.value > 0);
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

/** The plate as printed, with characters the reader doubts underlined (their probability on hover). */
export function PlateNumber({ read, className = '' }: { read: PlateRead; className?: string }) {
  const slots = charSlots(read.plate);
  return (
    <span className={`whitespace-pre font-mono font-semibold tracking-wide ${className}`} aria-label={read.plate}>
      {[...read.plate].map((ch, i) => {
        const c = slots[i] === null ? null : read.chars[slots[i]!];
        return (
          <span
            key={i}
            aria-hidden="true"
            title={c ? `${ch}: ${pct(c.prob)} sure` : undefined}
            className={c && c.prob < DOUBT ? 'text-accent underline decoration-dotted decoration-2 underline-offset-4' : undefined}
          >
            {ch}
          </span>
        );
      })}
    </span>
  );
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      setTimeout(() => setDone(false), 1500);
    } catch {
      // Clipboard blocked (e.g. an insecure context): the number is still selectable.
    }
  };
  return (
    <button type="button" onClick={copy} className="shrink-0 rounded-md p-1.5 text-muted transition-colors hover:bg-surface hover:text-text" aria-label={done ? 'Copied' : `Copy ${text}`}>
      {done ? <Check className="size-4 text-accent" aria-hidden="true" /> : <Copy className="size-4" aria-hidden="true" />}
    </button>
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

      {live ? (
        boxes.some((b) => b.read?.text) && (
          <ul className="flex flex-col gap-1.5">
            {boxes.map((b, i) => b.read?.text && <li key={i}><PlateNumber read={b.read} className="text-xl" /></li>)}
          </ul>
        )
      ) : (
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
                  {b.read?.text ? (
                    <>
                      <div className="flex items-center justify-between gap-2">
                        <PlateNumber read={b.read} className="truncate text-lg" />
                        <CopyButton text={b.read.plate} />
                      </div>
                      <p className="font-mono text-[11px] text-faint">
                        {b.read.format === 'ghana' ? 'Ghana format' : 'Other format'} · read {pct(b.read.confidence)} · found {pct(b.score)}
                      </p>
                    </>
                  ) : (
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-sm font-medium">Plate {i + 1}</span>
                      <span className="font-mono text-xs tabular-nums text-muted">found {pct(b.score)}</span>
                    </div>
                  )}
                </div>
              </m.li>
            ))}
          </AnimatePresence>
        </ul>
      )}

      {timing && <TimingBar timing={timing} />}
      {!live && boxes.some((b) => b.read?.chars.some((c) => c.prob < DOUBT)) && (
        <p className="text-xs leading-relaxed text-muted">
          <span className="font-mono text-accent underline decoration-dotted decoration-2 underline-offset-4">Underlined</span> characters are uncertain. Hover one to see how sure the reader is.
        </p>
      )}
    </div>
  );
}
