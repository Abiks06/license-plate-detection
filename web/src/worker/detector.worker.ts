/// <reference lib="webworker" />
// Runs the detector off the main thread so the UI and webcam preview stay smooth.

import * as ort from 'onnxruntime-web/wasm';
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { decode, letterbox, nms } from '../lib/postprocess';
import type { VariantName, WorkerRequest, WorkerResponse } from '../lib/types';

declare const self: DedicatedWorkerGlobalScope;

ort.env.wasm.wasmPaths = { mjs: mjsUrl, wasm: wasmUrl };
// Threads need cross-origin isolation (COOP/COEP headers); fall back to one thread without it.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

type Loaded = { session: ort.InferenceSession; imgsz: number; input: Float32Array; canvas: OffscreenCanvas };
const sessions = new Map<VariantName, Loaded>();
let active: VariantName | null = null;

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

/** Builds a session from bytes the page already downloaded (so the page's preloads are used once). */
async function load(variant: VariantName, model: ArrayBuffer, wasm: ArrayBuffer | undefined, imgsz: number) {
  // The runtime binary arrives with the first model; ORT compiles it on first session creation.
  if (wasm && !ort.env.wasm.wasmBinary) ort.env.wasm.wasmBinary = wasm;
  if (!sessions.has(variant)) {
    const session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    const loaded: Loaded = { session, imgsz, input: new Float32Array(3 * imgsz * imgsz), canvas: new OffscreenCanvas(imgsz, imgsz) };
    sessions.set(variant, loaded);
    // The first run compiles kernels; do it now so the first real detection is fast.
    const t = performance.now();
    await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', loaded.input, [1, 3, imgsz, imgsz]) });
    active = variant;
    post({ type: 'ready', variant, backend: 'wasm', threads: ort.env.wasm.numThreads ?? 1, warmup: performance.now() - t });
    return;
  }
  active = variant;
  post({ type: 'ready', variant, backend: 'wasm', threads: ort.env.wasm.numThreads ?? 1, warmup: 0 });
}

async function detect(id: number, image: ImageBitmap, threshold: number, iou: number) {
  const model = active && sessions.get(active);
  if (!model) throw new Error('Model is not loaded yet');
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

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') await load(msg.variant, msg.model, msg.wasm, msg.imgsz);
    else await detect(msg.id, msg.image, msg.threshold, msg.iou);
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err), id: msg.type === 'detect' ? msg.id : undefined });
    if (msg.type === 'detect') msg.image.close();
  }
};
