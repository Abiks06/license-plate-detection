import type { PlateRead } from './plateText';

/** A detected plate in source-image pixels, with what the reader made of it. */
export type Box = { x: number; y: number; w: number; h: number; score: number; read?: PlateRead };

export type Metrics = { mAP50: number; mAP50_95: number; precision: number; recall: number };

export type VariantName = 'fast' | 'accurate';

/** One entry of public/model/models.json, written by training/scripts/export.py. */
export type VariantMeta = {
  file: string;
  bytes: number;
  architecture: string;
  imgsz: number;
  classes: string[];
  // ghana: an outside test set (training/scripts/evaluate_external.py), never trained on.
  metrics: { all: Metrics; inTheWild?: Metrics; ghana?: Metrics };
  training: { epochs: number; imgsz: number; fraction: number; device: string; openImages: boolean };
  /** The plate reader (OCR) paired with this detector. */
  reader?: ReaderMeta;
};

export type ReaderMeta = {
  file: string;
  bytes: number;
  name: string;
  source: string;
  license: string;
  slots: number;
  width: number;
  height: number;
  /** Crop margin around each detected box, as a fraction of its size. */
  pad: number;
  /**
   * Exact-match rates on hand-labelled Ghanaian plates (training/scripts/evaluate_ocr.py): the
   * reader on labelled boxes, and end to end with this variant's detector finding the plates.
   */
  ghana?: { plates: number; exact: number; endToEnd: number };
};

export type Manifest = {
  variants: Partial<Record<VariantName, VariantMeta>>;
  datasets: { name: string; license: string; url: string }[];
};

export type Timing = { preprocess: number; inference: number; postprocess: number; read: number; total: number };

export type WorkerRequest =
  // Bytes are fetched on the page (which consumes the <link rel=preload>s) and transferred, not copied.
  | { type: 'load'; variant: VariantName; model: ArrayBuffer; runtime?: { backend: 'webgpu' | 'wasm'; wasm: ArrayBuffer }; imgsz: number }
  // A variant's plate reader, sent after its detector so it never delays the first detection.
  | { type: 'reader'; variant: VariantName; model: ArrayBuffer; slots: number; width: number; height: number; pad: number }
  | { type: 'detect'; id: number; image: ImageBitmap; threshold: number; iou: number };

export type WorkerResponse =
  | { type: 'ready'; variant: VariantName; backend: string; threads: number; warmup: number }
  | { type: 'result'; id: number; boxes: Box[]; timing: Timing; width: number; height: number }
  | { type: 'error'; message: string; id?: number };
