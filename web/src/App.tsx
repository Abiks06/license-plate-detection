import { m } from "motion/react";
import { Code, Cpu, ImageIcon, Lock, ScanSearch, Sparkles, Video } from 'lucide-react';
import { useState } from 'react';
import { HeroDemo } from './components/HeroDemo';
import { ImageMode } from './components/ImageMode';
import { WebcamMode } from './components/WebcamMode';
import { formatBytes, formatMs, pct } from './lib/image';
import type { Manifest, VariantName } from './lib/types';
import { useDetector, type DetectorState } from './lib/useDetector';

const ease = [0.16, 1, 0.3, 1] as const;
const REPO = 'https://github.com/Abiks06/license-plate-detection';
const PORTFOLIO = 'https://ayomide-abikoye.vercel.app';

function Status({ state }: { state: DetectorState }) {
  const text =
    state.status === 'ready'
      ? state.backend === 'webgpu'
        ? 'Ready · WebGPU'
        : `Ready · WebAssembly · ${state.threads} thread${state.threads === 1 ? '' : 's'}${state.upgrading ? ' · tuning GPU…' : ''}`
      : state.status === 'loading'
        ? `Loading model ${Math.round(state.progress * 100)}%`
        : state.status === 'error'
          ? 'Model failed to load'
          : 'Starting';
  const tone = state.status === 'ready' ? 'bg-emerald-400' : state.status === 'error' ? 'bg-danger' : 'bg-accent';
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1 font-mono text-xs text-muted" role="status" aria-live="polite">
      <span className="relative flex size-2">
        {state.status !== 'ready' && state.status !== 'error' && <span className={`absolute inline-flex size-full animate-ping rounded-full ${tone} opacity-60 motion-reduce:hidden`} />}
        <span className={`relative inline-flex size-2 rounded-full ${tone}`} />
      </span>
      {text}
      {state.status === 'loading' && (
        <span className="h-1 w-16 overflow-hidden rounded-full bg-surface-2">
          <m.span className="block h-full bg-accent" animate={{ width: `${state.progress * 100}%` }} transition={{ duration: 0.2 }} />
        </span>
      )}
    </span>
  );
}

function Segmented<T extends string>({
  id,
  value,
  options,
  onChange,
  label,
}: {
  id: string;
  value: T;
  options: { value: T; label: React.ReactNode; hint?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-full border border-line bg-surface p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`relative flex items-center gap-2 rounded-full px-3.5 py-1.5 text-sm transition-colors duration-200 ${value === o.value ? 'text-on-accent' : 'text-muted hover:text-text'}`}
        >
          {value === o.value && <m.span layoutId={`${id}-pill`} className="absolute inset-0 rounded-full bg-accent" transition={{ type: 'spring', stiffness: 420, damping: 34 }} />}
          <span className="relative flex items-center gap-1.5">{o.label}</span>
          {o.hint && <span className={`relative font-mono text-[10px] ${value === o.value ? 'text-on-accent' : 'text-faint'}`}>{o.hint}</span>}
        </button>
      ))}
    </div>
  );
}

function isPreview(m: Manifest | null) {
  const v = m?.variants.fast ?? m?.variants.accurate;
  return !!v && (v.training.epochs < 10 || v.training.fraction < 0.5);
}

function Metrics({ manifest, variant }: { manifest: Manifest; variant: VariantName }) {
  const meta = manifest.variants[variant];
  if (!meta) return null;
  const score = meta.metrics.inTheWild ?? meta.metrics.all;
  const items = [
    { label: 'mAP@50', value: pct(score.mAP50), note: meta.metrics.inTheWild ? 'in-the-wild test set' : 'held-out test set' },
    { label: 'Precision', value: pct(score.precision), note: 'boxes that are real plates' },
    { label: 'Recall', value: pct(score.recall), note: 'plates that get found' },
    { label: 'Model size', value: formatBytes(meta.bytes), note: `${meta.architecture} · ${meta.imgsz}px input` },
  ];
  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-4">
      {items.map((it, i) => (
        <m.div
          key={it.label}
          className="bg-surface p-5"
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.06 * i, duration: 0.5, ease }}
        >
          <dt className="font-mono text-[11px] uppercase tracking-wider text-faint">{it.label}</dt>
          <dd className="mt-2 text-2xl font-semibold tracking-tight tabular-nums">{it.value}</dd>
          <dd className="mt-1 text-xs text-muted">{it.note}</dd>
        </m.div>
      ))}
    </dl>
  );
}

/** Both shipped models side by side, so the size/accuracy trade-off is visible rather than asserted. */
function Comparison({ manifest, active }: { manifest: Manifest; active: VariantName }) {
  const rows = (['fast', 'accurate'] as const).flatMap((v) => (manifest.variants[v] ? [[v, manifest.variants[v]!] as const] : []));
  if (rows.length < 2) return null;
  return (
    <div className="mt-6 overflow-x-auto rounded-2xl border border-line">
      <table className="w-full min-w-[520px] text-left text-sm">
        <caption className="sr-only">Fast and Accurate models compared on held-out test images</caption>
        <thead className="bg-surface-2/60 font-mono text-[11px] uppercase tracking-wider text-muted">
          <tr>
            <th scope="col" className="px-4 py-3 font-medium">Model</th>
            <th scope="col" className="px-4 py-3 font-medium">mAP@50 in the wild</th>
            <th scope="col" className="px-4 py-3 font-medium">mAP@50–95</th>
            <th scope="col" className="px-4 py-3 font-medium">Recall</th>
            <th scope="col" className="px-4 py-3 font-medium">Download</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line bg-surface">
          {rows.map(([v, meta]) => {
            const s = meta.metrics.inTheWild ?? meta.metrics.all;
            return (
              <tr key={v} className={v === active ? 'bg-accent-soft' : undefined}>
                <th scope="row" className="px-4 py-3 font-medium">
                  {v === 'fast' ? 'Fast' : 'Accurate'} <span className="font-mono text-xs font-normal text-muted">{meta.architecture} · {meta.imgsz}px</span>
                </th>
                <td className="px-4 py-3 font-mono tabular-nums">{pct(s.mAP50)}</td>
                <td className="px-4 py-3 font-mono tabular-nums">{pct(s.mAP50_95)}</td>
                <td className="px-4 py-3 font-mono tabular-nums">{pct(s.recall)}</td>
                <td className="px-4 py-3 font-mono tabular-nums">{formatBytes(meta.bytes)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const STEPS = [
  { icon: ImageIcon, title: 'Letterbox', body: 'Your image is scaled into a square model input on a canvas, padded with grey so nothing is stretched.' },
  { icon: Cpu, title: 'YOLO11', body: 'A convolutional neural network fine-tuned on thousands of labelled plates scores every region of the image in one pass.' },
  { icon: ScanSearch, title: 'Decode + NMS', body: 'Candidate boxes above your confidence threshold are mapped back to the photo; overlapping duplicates are suppressed.' },
  { icon: Lock, title: 'Stays on device', body: 'All of it runs in a Web Worker, on your GPU via WebGPU where available or on the CPU via WebAssembly. No upload, no server, no tracking: close the tab and it’s gone.' },
];

export default function App() {
  const { manifest, state, detect, switchVariant } = useDetector();
  const [mode, setMode] = useState<'image' | 'webcam'>('image');
  const [threshold, setThreshold] = useState(0.35);
  const variant: VariantName = state.status === 'ready' || state.status === 'loading' ? state.variant : manifest?.variants.fast ? 'fast' : 'accurate';
  const ready = state.status === 'ready' ? `${state.variant}` : null;
  const variants = (['fast', 'accurate'] as const).filter((v) => manifest?.variants[v]);

  return (
    <div className="min-h-svh">
      <header className="sticky top-0 z-30 border-b border-line/70 bg-bg/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
          <a href="#top" className="flex items-center gap-2 font-semibold tracking-tight">
            <img src="/favicon.svg" alt="" className="size-6" /> PlateSight
          </a>
          <nav className="flex items-center gap-5 text-sm text-muted">
            <a href="#how" className="hidden transition-colors hover:text-text sm:block">How it works</a>
            <a href={REPO} aria-label="Source code on GitHub" className="flex items-center gap-1.5 transition-colors hover:text-text">
              <Code className="size-4" aria-hidden="true" /> <span className="hidden sm:inline">Source</span>
            </a>
          </nav>
        </div>
      </header>

      <main id="top">
        <section className="relative overflow-hidden">
          <div className="grid-bg pointer-events-none absolute inset-0" aria-hidden="true" />
          <div className="relative mx-auto grid max-w-6xl items-center gap-12 px-4 pb-10 pt-16 sm:px-6 sm:pt-24 lg:grid-cols-[minmax(0,1fr)_420px]">
            <div>
            {/* Hero animates with CSS (see index.css) so it never waits for JavaScript to show. */}
            <div className="anim-fade-up">
              <Status state={state} />
            </div>
            <h1 className="mt-6 max-w-3xl text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
              {['Find license plates', 'in any photo.'].map((line, i) => (
                <span key={line} className="block overflow-hidden pb-1">
                  <span className={`anim-rise block ${i === 1 ? 'text-accent' : ''}`} style={{ animationDelay: `${0.1 + i * 0.1}s` }}>
                    {line}
                  </span>
                </span>
              ))}
            </h1>
            <p className="anim-fade-up mt-5 max-w-xl text-lg leading-relaxed text-muted" style={{ animationDelay: '0.3s' }}>
              A YOLO11 neural network that runs entirely in your browser. Detect plates in photos or live from your camera, then redact them in one click. Nothing is ever uploaded.
            </p>
            {isPreview(manifest) && (
              <p className="anim-fade-up mt-6 inline-flex items-center gap-2 rounded-lg border border-accent/40 bg-accent-soft px-3 py-2 text-sm">
                <Sparkles className="size-4 text-accent" aria-hidden="true" /> Preview model: the fully trained detectors are on their way.
              </p>
            )}
            </div>
            <div className="anim-fade-up hidden lg:block" style={{ animationDelay: '0.25s' }}>
              <HeroDemo />
            </div>
          </div>
        </section>

        <section aria-label="Detector" className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="mb-5 flex flex-wrap items-center gap-3">
            <Segmented
              id="mode"
              label="Input"
              value={mode}
              onChange={setMode}
              options={[
                { value: 'image', label: <><ImageIcon className="size-4" aria-hidden="true" /> Image</> },
                { value: 'webcam', label: <><Video className="size-4" aria-hidden="true" /> Live camera</> },
              ]}
            />
            {variants.length > 1 && manifest && (
              <Segmented
                id="variant"
                label="Model"
                value={variant}
                onChange={(v) => switchVariant(v)}
                options={variants.map((v) => ({ value: v, label: v === 'fast' ? 'Fast' : 'Accurate', hint: formatBytes(manifest.variants[v]!.bytes) }))}
              />
            )}
            <label className="ml-auto flex items-center gap-3 rounded-full border border-line bg-surface px-4 py-1.5 text-sm text-muted">
              Confidence
              <input type="range" min={0.1} max={0.9} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-28" aria-valuetext={pct(threshold)} />
              <span className="w-9 font-mono text-xs tabular-nums text-text">{pct(threshold)}</span>
            </label>
          </div>

          {state.status === 'error' ? (
            <p className="rounded-2xl border border-danger/40 p-6 text-sm text-danger" role="alert">
              The model couldn’t load: {state.message}. Try reloading the page.
            </p>
          ) : mode === 'image' ? (
            <ImageMode ready={ready} threshold={threshold} detect={detect} />
          ) : (
            <WebcamMode ready={ready} threshold={threshold} detect={detect} />
          )}
          {state.status === 'ready' && state.warmup > 0 && (
            <p className="mt-3 font-mono text-[11px] text-faint">Model warmed up in {formatMs(state.warmup)}.</p>
          )}
        </section>

        <section id="how" className="mx-auto max-w-6xl scroll-mt-20 px-4 py-24 sm:px-6">
          <m.h2 className="text-3xl font-semibold tracking-tight sm:text-4xl" initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            How it works
          </m.h2>
          <ol className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {STEPS.map((s, i) => (
              <m.li
                key={s.title}
                className="group relative rounded-2xl border border-line bg-surface p-5"
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-10% 0px' }}
                transition={{ delay: 0.08 * i, duration: 0.6, ease }}
              >
                <span className="font-mono text-xs text-faint">0{i + 1}</span>
                <s.icon className="mt-4 size-5 text-accent transition-transform duration-300 ease-out-expo group-hover:-translate-y-0.5" aria-hidden="true" />
                <h3 className="mt-3 font-medium">{s.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-muted">{s.body}</p>
              </m.li>
            ))}
          </ol>
          {manifest && (
            <div className="mt-10">
              <p className="mb-3 font-mono text-[11px] uppercase tracking-wider text-faint">
                {variant === 'fast' ? 'Fast' : 'Accurate'} model, measured on images it never trained on
              </p>
              <Metrics manifest={manifest} variant={variant} />
              <Comparison manifest={manifest} active={variant} />
            </div>
          )}
        </section>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 text-xs text-faint sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <p>
            Built by <a href={PORTFOLIO} className="text-muted underline underline-offset-4 hover:text-text">Ayomide Abikoye</a>. Code AGPL-3.0.
          </p>
          <p>
            Data:{' '}
            {(manifest?.datasets ?? []).map((d, i) => (
              <span key={d.url}>
                {i > 0 && ' · '}
                <a href={d.url} className="underline underline-offset-4 hover:text-text">{d.name}</a> ({d.license})
              </span>
            ))}
          </p>
        </div>
      </footer>
    </div>
  );
}
