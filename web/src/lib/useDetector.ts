import { useCallback, useEffect, useRef, useState } from 'react';
import type { Box, Manifest, Timing, VariantName, WorkerResponse } from './types';

export type DetectorState =
  | { status: 'loading-manifest' }
  | { status: 'loading'; variant: VariantName; progress: number }
  | { status: 'ready'; variant: VariantName; backend: string; threads: number; warmup: number }
  | { status: 'error'; message: string };

export type Detection = { boxes: Box[]; timing: Timing; width: number; height: number };

const MODEL_BASE = `${import.meta.env.BASE_URL}model/`;

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
      if (msg.type === 'progress') {
        setState({ status: 'loading', variant: msg.variant, progress: msg.total ? msg.loaded / msg.total : 0 });
      } else if (msg.type === 'ready') {
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

  const load = useCallback(
    (variant: VariantName, m: Manifest) => {
      const meta = m.variants[variant];
      if (!meta || !worker.current) return;
      setState({ status: 'loading', variant, progress: 0 });
      worker.current.postMessage({ type: 'load', variant, url: MODEL_BASE + meta.file, imgsz: meta.imgsz });
    },
    [],
  );

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
