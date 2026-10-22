import { useCallback, useEffect, useRef, useState } from 'react';
import { runtimes, type Backend, type Runtime } from './runtime';
import type { Box, Manifest, Timing, VariantName, WorkerResponse } from './types';

export type DetectorState =
  | { status: 'loading-manifest' }
  | { status: 'loading'; variant: VariantName; progress: number }
  | { status: 'ready'; variant: VariantName; backend: Backend; threads: number; warmup: number; upgrading: boolean }
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

type Kind = 'cpu' | 'gpu';
type Slot = {
  kind: Kind;
  worker: Worker;
  runtime: Promise<Runtime | null>;
  runtimeSent: boolean;
  sent: Set<VariantName>;
  inflight: number;
  retiring: boolean;
  info?: { variant: VariantName; backend: Backend; threads: number; warmup: number };
};

/**
 * Owns the detector workers. The CPU worker answers within a second or so; on devices
 * with a hardware GPU a second, WebGPU worker warms up in the background (its first run
 * compiles shaders) and takes over once it's ready for the selected model. Detections
 * always go to the active worker, so the switch is invisible apart from the speed-up.
 */
export function useDetector() {
  const slots = useRef<Partial<Record<Kind, Slot>>>({});
  const active = useRef<Kind | null>(null);
  const desired = useRef<VariantName | null>(null);
  const modelBytes = useRef(new Map<VariantName, Promise<ArrayBuffer>>());
  const manifestRef = useRef<Manifest | null>(null);
  const pending = useRef(new Map<number, { resolve: (d: Detection) => void; reject: (e: Error) => void; slot: Slot }>());
  const nextId = useRef(1);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [state, setState] = useState<DetectorState>({ status: 'loading-manifest' });
  // Starts the WebGPU worker once; set in the mount effect (it needs spawn).
  const startGpu = useRef<() => void>(() => {});

  const publish = useCallback(() => {
    const slot = active.current && slots.current[active.current];
    if (!slot?.info || slot.info.variant !== desired.current) return;
    const gpu = slots.current.gpu;
    setState({ status: 'ready', ...slot.info, upgrading: slot.kind === 'cpu' && !!gpu && !gpu.retiring });
  }, []);

  const retire = useCallback((slot: Slot) => {
    slot.retiring = true;
    if (slot.inflight === 0) {
      slot.worker.terminate();
      delete slots.current[slot.kind];
    }
  }, []);

  /** Sends a variant to a worker: the model bytes (a copy) the first time, a switch afterwards. */
  const send = useCallback(async (slot: Slot, variant: VariantName) => {
    const meta = manifestRef.current?.variants[variant];
    if (!meta || slot.retiring) return;
    if (slot.sent.has(variant)) {
      slot.worker.postMessage({ type: 'load', variant, model: new ArrayBuffer(0), imgsz: meta.imgsz });
      return;
    }
    const download = modelBytes.current.get(variant);
    if (!download) return; // load() sends it to every live worker once downloaded
    slot.sent.add(variant);
    const [bytes, runtime] = await Promise.all([download, slot.runtime]);
    if (!runtime || slot.retiring) return;
    const model = bytes.slice(0);
    const first = !slot.runtimeSent;
    slot.runtimeSent = true;
    slot.worker.postMessage(
      { type: 'load', variant, model, runtime: first ? runtime : undefined, imgsz: meta.imgsz },
      first ? [model, runtime.wasm] : [model],
    );
  }, []);

  const onMessage = useCallback(
    (slot: Slot, msg: WorkerResponse) => {
      if (msg.type === 'ready') {
        slot.info = { variant: msg.variant, backend: msg.backend as Backend, threads: msg.threads, warmup: msg.warmup };
        if (slot.kind === 'gpu') {
          if (msg.backend !== 'webgpu') return retire(slot); // GPU session fell back to CPU: no gain
          if (msg.variant === desired.current) {
            active.current = 'gpu';
            const cpu = slots.current.cpu;
            if (cpu) retire(cpu);
          }
        } else if (active.current !== 'gpu') {
          active.current = 'cpu';
          startGpu.current();
        }
        publish();
        return;
      }
      if (msg.type === 'result' || msg.id !== undefined) {
        const req = pending.current.get(msg.id!);
        if (req) {
          pending.current.delete(msg.id!);
          req.slot.inflight -= 1;
          if (msg.type === 'result') req.resolve(msg);
          else req.reject(new Error(msg.message));
          if (req.slot.retiring) retire(req.slot);
        }
        return;
      }
      // A load error: the GPU worker is optional, the CPU worker is not.
      if (slot.kind === 'gpu') {
        console.info('WebGPU worker failed; staying on WebAssembly:', msg.message);
        retire(slot);
        publish();
      } else {
        setState({ status: 'error', message: msg.message });
      }
    },
    [publish, retire],
  );

  const spawn = useCallback(
    (kind: Kind, runtime: Promise<Runtime | null>) => {
      const worker = new Worker(new URL('../worker/detector.worker.ts', import.meta.url), { type: 'module' });
      const slot: Slot = { kind, worker, runtime, runtimeSent: false, sent: new Set(), inflight: 0, retiring: false };
      worker.onmessage = (e: MessageEvent<WorkerResponse>) => onMessage(slot, e.data);
      slots.current[kind] = slot;
      if (desired.current) send(slot, desired.current);
      return slot;
    },
    [onMessage, send],
  );

  const load = useCallback(
    async (variant: VariantName) => {
      const meta = manifestRef.current?.variants[variant];
      if (!meta) return;
      desired.current = variant;
      const live = () => Object.values(slots.current).filter((s): s is Slot => !!s && !s.retiring);
      if (!modelBytes.current.has(variant)) {
        setState({ status: 'loading', variant, progress: 0 });
        modelBytes.current.set(variant, fetchWithProgress(MODEL_BASE + meta.file, (p) => setState({ status: 'loading', variant, progress: p })));
      } else if (!live().some((s) => s.info?.variant === variant)) {
        setState({ status: 'loading', variant, progress: 1 });
      }
      try {
        await modelBytes.current.get(variant);
      } catch (e) {
        modelBytes.current.delete(variant);
        setState({ status: 'error', message: e instanceof Error ? e.message : String(e) });
        return;
      }
      // Re-read the workers: the GPU one may have started while the model downloaded.
      if (desired.current === variant) for (const slot of live()) send(slot, variant);
    },
    [send],
  );

  // Start the workers, fetch the manifest, then load the Fast model: it's smallest, so the app is usable soonest.
  useEffect(() => {
    let cancelled = false;
    let gpuStarted = false;
    // The GPU worker starts after the CPU worker is ready (or straight away when there is
    // no CPU worker), so its 6 MB runtime never delays the first usable moment.
    startGpu.current = () => {
      if (gpuStarted || cancelled) return;
      gpuStarted = true;
      runtimes
        .gpu()
        .then((rt) => {
          if (cancelled || !rt) return;
          spawn('gpu', Promise.resolve(rt));
          publish(); // show "tuning GPU…" while it warms up
        })
        .catch(() => {});
    };
    if (runtimes.cpu) spawn('cpu', runtimes.cpu);
    else startGpu.current();
    fetch(`${MODEL_BASE}models.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`models.json: ${r.status}`))))
      .then((m: Manifest) => {
        if (cancelled) return;
        manifestRef.current = m;
        setManifest(m);
        load(m.variants.fast ? 'fast' : 'accurate');
      })
      .catch((e: Error) => !cancelled && setState({ status: 'error', message: e.message }));
    const all = slots.current;
    const requests = pending.current;
    return () => {
      cancelled = true;
      for (const slot of Object.values(all)) slot?.worker.terminate();
      for (const k of Object.keys(all) as Kind[]) delete all[k];
      requests.clear();
    };
  }, [spawn, load, publish]);

  /** Runs one detection on the active worker. The bitmap is transferred to it and closed there. */
  const detect = useCallback((image: ImageBitmap, threshold: number, iou = 0.45) => {
    return new Promise<Detection>((resolve, reject) => {
      const slot = active.current && slots.current[active.current];
      if (!slot) {
        image.close();
        return reject(new Error('Detector not ready'));
      }
      const id = nextId.current++;
      slot.inflight += 1;
      pending.current.set(id, { resolve, reject, slot });
      slot.worker.postMessage({ type: 'detect', id, image, threshold, iou }, [image]);
    });
  }, []);

  const switchVariant = useCallback((variant: VariantName) => load(variant), [load]);

  return { manifest, state, detect, switchVariant };
}
