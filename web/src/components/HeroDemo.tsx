import { AnimatePresence, m, useReducedMotion } from "motion/react";
import { useEffect, useState } from 'react';
import type { Box } from '../lib/types';
import { Overlay } from './Overlay';

type Sample = { id: number; src: string; width: number; height: number; truth: Omit<Box, 'score'>[] };

const BASE = import.meta.env.BASE_URL;
const SCAN_MS = 1100;
const HOLD_MS = 3200;
// Samples with one clear plate read best at hero size.
const ORDER = [2, 4, 6, 1];

/** Looping "scan, then find" demo over test images, using their ground-truth labels. */
export function HeroDemo() {
  const reduced = useReducedMotion();
  const [samples, setSamples] = useState<Sample[]>([]);
  const [i, setI] = useState(0);
  const [found, setFound] = useState(false);

  useEffect(() => {
    fetch(`${BASE}samples/samples.json`)
      .then((r) => r.json())
      .then((d: { samples: Sample[] }) => setSamples(ORDER.map((id) => d.samples.find((s) => s.id === id)).filter((s): s is Sample => !!s)))
      .catch(() => setSamples([]));
  }, []);

  useEffect(() => {
    if (!samples.length || reduced) return;
    const scan = setTimeout(() => setFound(true), SCAN_MS);
    const next = setTimeout(() => {
      setFound(false);
      setI((n) => (n + 1) % samples.length);
    }, SCAN_MS + HOLD_MS);
    return () => {
      clearTimeout(scan);
      clearTimeout(next);
    };
  }, [i, samples, reduced]);

  // Reduced motion: no loop, just the first image with its plates.
  const showBoxes = reduced || found;
  const s = samples[i];
  if (!s) return <div className="aspect-[4/3] rounded-2xl border border-line bg-surface" />;

  return (
    <figure>
      <div className="relative overflow-hidden rounded-2xl border border-line bg-surface shadow-2xl shadow-black/30" style={{ aspectRatio: `${s.width} / ${s.height}` }}>
        <AnimatePresence initial={false}>
          <m.div key={s.id} className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.6 }}>
            <img src={BASE + s.src} alt="" className="h-full w-full object-cover" decoding="async" />
            <Overlay width={s.width} height={s.height} boxes={showBoxes ? s.truth.map((b) => ({ ...b, score: 1 })) : []} scanning={!showBoxes} showScore={false} />
          </m.div>
        </AnimatePresence>
        <div className="absolute bottom-3 left-3 flex gap-1.5" aria-hidden="true">
          {samples.map((x, n) => (
            <span key={x.id} className={`h-1 rounded-full transition-all duration-500 ${n === i ? 'w-6 bg-accent' : 'w-2 bg-white/40'}`} />
          ))}
        </div>
      </div>
      <figcaption className="mt-3 font-mono text-[11px] text-faint">Test-set images with their labelled plates. Try the real detector below.</figcaption>
    </figure>
  );
}
