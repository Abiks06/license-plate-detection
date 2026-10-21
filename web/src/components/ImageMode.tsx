import { AnimatePresence, m } from "motion/react";
import { Download, EyeOff, ImageUp, RotateCcw, ScanLine } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cropPlates, download, redact } from '../lib/image';
import type { Detection } from '../lib/useDetector';
import { Overlay } from './Overlay';
import { Results } from './Results';

const ease = [0.16, 1, 0.3, 1] as const;

type Source = { blob: Blob; url: string; width: number; height: number; name: string };
type Sample = { id: number; src: string; thumb: string; width: number; height: number };

const BASE = import.meta.env.BASE_URL;

function useSamples() {
  const [samples, setSamples] = useState<Sample[]>([]);
  useEffect(() => {
    fetch(`${BASE}samples/samples.json`)
      .then((r) => r.json())
      .then((d) => setSamples(d.samples))
      .catch(() => setSamples([]));
  }, []);
  return samples;
}

async function toSource(blob: Blob, name: string): Promise<Source> {
  const bitmap = await createImageBitmap(blob);
  const source = { blob, url: URL.createObjectURL(blob), width: bitmap.width, height: bitmap.height, name };
  bitmap.close();
  return source;
}

export function ImageMode({
  ready,
  threshold,
  detect,
}: {
  ready: string | null;
  threshold: number;
  detect: (image: ImageBitmap, threshold: number) => Promise<Detection>;
}) {
  const samples = useSamples();
  const [source, setSource] = useState<Source | null>(null);
  const [result, setResult] = useState<Detection | null>(null);
  const [crops, setCrops] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [redacted, setRedacted] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const run = useRef(0);

  const open = useCallback(async (blob: Blob, name: string) => {
    if (!blob.type.startsWith('image/')) return setError('That file isn’t an image.');
    setError(null);
    setRedacted(null);
    setResult(null);
    setCrops([]);
    setSource((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return null;
    });
    setSource(await toSource(blob, name));
  }, []);

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith('image/'));
      if (file) open(file, 'pasted-image');
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [open]);

  // (Re)detect whenever the image, the threshold or the loaded model changes. Stale runs are dropped.
  useEffect(() => {
    if (!source || !ready) return;
    const id = ++run.current;
    const timer = setTimeout(async () => {
      setBusy(true);
      try {
        const bitmap = await createImageBitmap(source.blob);
        const res = await detect(bitmap, threshold);
        if (id !== run.current) return;
        setResult(res);
        const img = await createImageBitmap(source.blob);
        setCrops(cropPlates(img, source.width, source.height, res.boxes));
        setRedacted((r) => (r ? redact(img, source.width, source.height, res.boxes).toDataURL('image/jpeg', 0.9) : null));
        img.close();
      } catch (e) {
        if (id === run.current) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (id === run.current) setBusy(false);
      }
    }, 120);
    return () => clearTimeout(timer);
  }, [source, ready, threshold, detect]);

  const toggleRedact = async () => {
    if (!source || !result) return;
    if (redacted) return setRedacted(null);
    const img = await createImageBitmap(source.blob);
    setRedacted(redact(img, source.width, source.height, result.boxes).toDataURL('image/jpeg', 0.9));
    img.close();
  };

  const save = async () => {
    if (!source || !result) return;
    const img = await createImageBitmap(source.blob);
    download(redact(img, source.width, source.height, result.boxes), `${source.name.replace(/\.[^.]+$/, '')}-redacted.jpg`);
    img.close();
  };

  const loadSample = async (s: Sample) => {
    const blob = await (await fetch(BASE + s.src)).blob();
    open(blob, `sample-${s.id}.jpg`);
  };

  const aspect = useMemo(() => (source ? `${source.width} / ${source.height}` : '16 / 10'), [source]);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div>
        <div
          className={`relative overflow-hidden rounded-2xl border bg-surface-2/40 transition-colors duration-200 ${dragging ? 'border-accent' : 'border-line'}`}
          style={{ aspectRatio: aspect, maxHeight: '70vh' }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) open(file, file.name);
          }}
        >
          <AnimatePresence mode="wait" initial={false}>
            {source ? (
              <m.div
                key={source.url}
                className="absolute inset-0"
                initial={{ opacity: 0, scale: 0.985 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4, ease }}
              >
                <img src={redacted ?? source.url} alt="Your image" className="h-full w-full object-contain" draggable={false} />
                {!redacted && result && (
                  <Overlay width={source.width} height={source.height} boxes={result.boxes} highlighted={highlight} scanning={busy} />
                )}
                {!result && busy && <Overlay width={source.width} height={source.height} boxes={[]} scanning />}
              </m.div>
            ) : (
              <m.button
                key="drop"
                type="button"
                onClick={() => input.current?.click()}
                className="group absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <m.span
                  className="grid size-14 place-items-center rounded-2xl border border-line-strong bg-surface text-accent"
                  animate={dragging ? { scale: 1.08, rotate: -4 } : { scale: 1, rotate: 0 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 22 }}
                >
                  <ImageUp className="size-6" aria-hidden="true" />
                </m.span>
                <span>
                  <span className="block text-base font-medium">Drop a photo, paste one, or browse</span>
                  <span className="mt-1 block text-sm text-muted">JPG, PNG or WebP · processed on your device, never uploaded</span>
                </span>
              </m.button>
            )}
          </AnimatePresence>
          <input
            ref={input}
            type="file"
            accept="image/*"
            className="sr-only"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) open(file, file.name);
              e.target.value = '';
            }}
          />
        </div>

        {error && <p className="mt-3 text-sm text-danger" role="alert">{error}</p>}

        <div className="mt-4 flex flex-wrap items-center gap-2">
          {source && (
            <>
              <button type="button" onClick={() => input.current?.click()} className="btn">
                <RotateCcw className="size-4" aria-hidden="true" /> New image
              </button>
              <button type="button" onClick={toggleRedact} disabled={!result?.boxes.length} aria-pressed={!!redacted} className="btn">
                {redacted ? <ScanLine className="size-4" aria-hidden="true" /> : <EyeOff className="size-4" aria-hidden="true" />}
                {redacted ? 'Show detections' : 'Redact plates'}
              </button>
              <button type="button" onClick={save} disabled={!result?.boxes.length} className="btn btn-primary">
                <Download className="size-4" aria-hidden="true" /> Download redacted
              </button>
            </>
          )}
        </div>

        {samples.length > 0 && (
          <div className="mt-8">
            <p className="font-mono text-[11px] uppercase tracking-wider text-faint">Or try a test image the model never saw</p>
            <ul className="mt-3 grid grid-cols-4 gap-2 sm:grid-cols-8">
              {samples.map((s, i) => (
                <m.li key={s.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.04 * i, duration: 0.4, ease }}>
                  <button
                    type="button"
                    onClick={() => loadSample(s)}
                    className="block aspect-square w-full overflow-hidden rounded-lg border border-line transition-all duration-200 hover:-translate-y-0.5 hover:border-accent active:scale-95"
                    aria-label={`Sample image ${s.id}`}
                  >
                    <img src={BASE + s.thumb} alt="" loading="lazy" decoding="async" className="h-full w-full object-cover" />
                  </button>
                </m.li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <aside className="rounded-2xl border border-line bg-surface p-5">
        {result ? (
          <Results boxes={result.boxes} crops={crops} timing={result.timing} onHover={setHighlight} />
        ) : (
          <div className="flex h-full min-h-48 flex-col justify-center text-sm text-muted">
            <span className="font-mono text-[11px] uppercase tracking-wider text-faint">Results</span>
            <p className="mt-2">{ready ? 'Choose an image to find its plates.' : 'Loading the model…'}</p>
          </div>
        )}
      </aside>
    </div>
  );
}
