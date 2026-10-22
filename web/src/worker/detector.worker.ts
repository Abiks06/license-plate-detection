/// <reference lib="webworker" />
// Runs the detector off the main thread so the UI and webcam preview stay smooth.

import cpuMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import gpuMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import { decode, letterbox, nms } from '../lib/postprocess';
import type { Backend } from '../lib/runtime';
import type { VariantName, WorkerRequest, WorkerResponse } from '../lib/types';

declare const self: DedicatedWorkerGlobalScope;

type Ort = typeof import('onnxruntime-web/wasm');
type Loaded = { session: import('onnxruntime-web/wasm').InferenceSession; imgsz: number; input: Float32Array; canvas: OffscreenCanvas };

// Only the build for the chosen backend is imported: the WebGPU one (which also has a
// CPU fallback) or the smaller CPU-only one. Set on the first load message.
let ort: Ort | null = null;
let requested: Backend = 'wasm';
const sessions = new Map<VariantName, Loaded & { backend: Backend }>();
let active: VariantName | null = null;

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);
const threads = () => (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1);

async function init(backend: Backend, wasm: ArrayBuffer) {
  ort = backend === 'webgpu' ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm');
  ort.env.wasm.wasmPaths = { mjs: backend === 'webgpu' ? gpuMjsUrl : cpuMjsUrl };
  ort.env.wasm.wasmBinary = wasm;
  // Threads need cross-origin isolation (COOP/COEP headers); fall back to one thread without it.
  ort.env.wasm.numThreads = threads();
  requested = backend;
}

/** Builds a session from bytes the page already downloaded (so the page's preloads are used once). */
async function load(variant: VariantName, model: ArrayBuffer, runtime: { backend: Backend; wasm: ArrayBuffer } | undefined, imgsz: number) {
  if (runtime && !ort) await init(runtime.backend, runtime.wasm);
  if (!ort) throw new Error('Runtime not initialised');
  const existing = sessions.get(variant);
  if (existing) {
    active = variant;
    post({ type: 'ready', variant, backend: existing.backend, threads: threads(), warmup: 0 });
    return;
  }

  const bytes = new Uint8Array(model);
  const options = { graphOptimizationLevel: 'all' as const };
  let session: Loaded['session'];
  let backend: Backend = requested;
  try {
    session = await ort.InferenceSession.create(bytes, { ...options, executionProviders: [requested] });
  } catch (err) {
    if (requested !== 'webgpu') throw err;
    // WebGPU can fail on some drivers; the same build runs on the CPU instead.
    console.info('WebGPU session failed, using WebAssembly:', err);
    session = await ort.InferenceSession.create(bytes, { ...options, executionProviders: ['wasm'] });
    backend = 'wasm';
  }
  const loaded = { session, imgsz, input: new Float32Array(3 * imgsz * imgsz), canvas: new OffscreenCanvas(imgsz, imgsz), backend };
  sessions.set(variant, loaded);
  // The first run compiles kernels (or GPU shaders); do it now so the first real detection is fast.
  const t = performance.now();
  await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', loaded.input, [1, 3, imgsz, imgsz]) });
  active = variant;
  post({ type: 'ready', variant, backend, threads: threads(), warmup: performance.now() - t });
}

async function detect(id: number, image: ImageBitmap, threshold: number, iou: number) {
  const model = active && sessions.get(active);
  if (!model || !ort) throw new Error('Model is not loaded yet');
  const { session, imgsz, input, canvas } = model;
  const t0 = performance.now();

  // Letterbox: fit the image into the square input, padding with YOLO's grey (114).
  const lb = letterbox(image.width, image.height, imgsz);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, imgsz, imgsz);
  ctx.drawImage(image, lb.padX, lb.padY, image.width * lb.scale, image.height * lb.scale);
  const { data } = ctx.getImageData(0, 0, imgsz, imgsz);
  // HWC RGBA bytes -> CHW float RGB in [0, 1].
  const plane = imgsz * imgsz;
  for (let p = 0, i = 0; p < plane; p++, i += 4) {
    input[p] = data[i] / 255;
    input[plane + p] = data[i + 1] / 255;
    input[2 * plane + p] = data[i + 2] / 255;
  }
  const t1 = performance.now();

  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, imgsz, imgsz]) });
  const tensor = out[session.outputNames[0]];
  const t2 = performance.now();

  const anchors = tensor.dims[2];
  const boxes = nms(decode(tensor.data as Float32Array, anchors, threshold, lb, image.width, image.height), iou);
  const t3 = performance.now();

  post({
    type: 'result',
    id,
    boxes,
    width: image.width,
    height: image.height,
    timing: { preprocess: t1 - t0, inference: t2 - t1, postprocess: t3 - t2, total: t3 - t0 },
  });
  image.close();
}

async function handle(msg: WorkerRequest) {
  try {
    if (msg.type === 'load') await load(msg.variant, msg.model, msg.runtime, msg.imgsz);
    else await detect(msg.id, msg.image, msg.threshold, msg.iou);
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err), id: msg.type === 'detect' ? msg.id : undefined });
    if (msg.type === 'detect') msg.image.close();
  }
}

// One message at a time, in order: a model switch arriving mid-load must not interleave with it.
let queue = Promise.resolve();
self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  queue = queue.then(() => handle(e.data));
};
