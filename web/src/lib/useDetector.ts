import { useCallback, useEffect, useRef, useState } from 'react';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import type { Box, Manifest, Timing, VariantName, WorkerResponse } from './types';

export type DetectorState =
  | { status: 'loading-manifest' }
  | { status: 'loading'; variant: VariantName; progress: number }
  | { status: 'ready'; variant: VariantName; backend: string; threads: number; warmup: number }
  | { status: 'error'; message: string };

export type Detection = { boxes: Box[]; timing: Timing; width: number; height: number };

const MODEL_BASE = `${import.meta.env.BASE_URL}model/`;

/** Downloads a file as an ArrayBuffer, reporting progress in [0, 1]. */
async function fetchWithProgress(url: string, onProgress: (p: number) => void) {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Model download failed (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (total) onProgress(loaded / total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.byteLength;
  }
  return bytes.buffer;
}

/** Owns the detector worker: loads the manifest and models, and runs detections. */
export function useDetector() {
  const worker = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, { resolve: (d: Detection) => void; reject: (e: Error) => void }>());
  const nextId = useRef(1);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [state, setState] = useState<DetectorState>({ status: 'loading-manifest' });

  useEffect(() => {
    const w = new Worker(new URL('../worker/detector.worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        setState({ status: 'ready', variant: msg.variant, backend: msg.backend, threads: msg.threads, warmup: msg.warmup });
      } else if (msg.type === 'result') {
        pending.current.get(msg.id)?.resolve(msg);
        pending.current.delete(msg.id);
      } else if (msg.id !== undefined) {
        pending.current.get(msg.id)?.reject(new Error(msg.message));
        pending.current.delete(msg.id);
      } else {
        setState({ status: 'error', message: msg.message });
      }
    };
    const requests = pending.current;
    return () => {
      w.terminate();
      requests.clear();
    };
  }, []);

  const sent = useRef(new Set<VariantName>());
  const runtime = useRef<Promise<ArrayBuffer> | null>(null);

  const load = useCallback(async (variant: VariantName, m: Manifest) => {
    const meta = m.variants[variant];
    const w = worker.current;
    if (!meta || !w) return;
    // Already in the worker: just switch to it.
    if (sent.current.has(variant)) {
      w.postMessage({ type: 'load', variant, model: new ArrayBuffer(0), imgsz: meta.imgsz });
      return;
    }
    setState({ status: 'loading', variant, progress: 0 });
    try {
      // Both requests are satisfied by index.html's preloads on first load. The runtime goes once.
      const first = !runtime.current;
      runtime.current ??= fetch(wasmUrl).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`runtime: ${r.status}`))));
      const [model, wasm] = await Promise.all([
        fetchWithProgress(MODEL_BASE + meta.file, (p) => setState({ status: 'loading', variant, progress: p })),
        runtime.current,
      ]);
      sent.current.add(variant);
      const transfer: Transferable[] = first ? [model, wasm] : [model];
      w.postMessage({ type: 'load', variant, model, wasm: first ? wasm : undefined, imgsz: meta.imgsz }, transfer);
    } catch (e) {
      setState({ status: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  // Fetch the manifest, then load the fast model first: it's smallest, so the app is usable soonest.
  useEffect(() => {
    let cancelled = false;
    fetch(`${MODEL_BASE}models.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`models.json: ${r.status}`))))
      .then((m: Manifest) => {
        if (cancelled) return;
        setManifest(m);
        const first: VariantName = m.variants.fast ? 'fast' : 'accurate';
        load(first, m);
      })
      .catch((e: Error) => !cancelled && setState({ status: 'error', message: e.message }));
    return () => {
      cancelled = true;
    };
  }, [load]);

  /** Runs one detection. The bitmap is transferred to the worker and closed there. */
  const detect = useCallback((image: ImageBitmap, threshold: number, iou = 0.45) => {
    return new Promise<Detection>((resolve, reject) => {
      if (!worker.current) return reject(new Error('Detector not started'));
      const id = nextId.current++;
      pending.current.set(id, { resolve, reject });
      worker.current.postMessage({ type: 'detect', id, image, threshold, iou }, [image]);
    });
  }, []);

  const switchVariant = useCallback((variant: VariantName) => manifest && load(variant, manifest), [load, manifest]);

  return { manifest, state, detect, switchVariant };
}
