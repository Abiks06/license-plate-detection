/** A detected plate in source-image pixels. */
export type Box = { x: number; y: number; w: number; h: number; score: number };

export type Metrics = { mAP50: number; mAP50_95: number; precision: number; recall: number };

export type VariantName = 'fast' | 'accurate';

/** One entry of public/model/models.json, written by training/scripts/export.py. */
export type VariantMeta = {
  file: string;
  bytes: number;
  architecture: string;
  imgsz: number;
  classes: string[];
  metrics: { all: Metrics; inTheWild?: Metrics };
  training: { epochs: number; imgsz: number; fraction: number; device: string; openImages: boolean };
};

export type Manifest = {
  variants: Partial<Record<VariantName, VariantMeta>>;
  datasets: { name: string; license: string; url: string }[];
};

export type Timing = { preprocess: number; inference: number; postprocess: number; total: number };

export type WorkerRequest =
  // Bytes are fetched on the page (which consumes the <link rel=preload>s) and transferred, not copied.
  | { type: 'load'; variant: VariantName; model: ArrayBuffer; runtime?: { backend: 'webgpu' | 'wasm'; wasm: ArrayBuffer }; imgsz: number }
  | { type: 'detect'; id: number; image: ImageBitmap; threshold: number; iou: number };

export type WorkerResponse =
  | { type: 'ready'; variant: VariantName; backend: string; threads: number; warmup: number }
  | { type: 'result'; id: number; boxes: Box[]; timing: Timing; width: number; height: number }
  | { type: 'error'; message: string; id?: number };
