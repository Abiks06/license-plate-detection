"""Measures how often the app reads a Ghanaian plate's number exactly right.

Uses the hand-transcribed plates in training/ocr/ghana-plates.csv (310 readable of 380, from the
public-domain Ghanaian License Plates dataset; blank = too blurry or cut off for a person to read).

Two measurements:
- reader: the plate reader on the labelled boxes, so only reading is tested.
- end to end: what the app does, detector -> crop -> reader, on the whole photo. A plate the
  detector misses counts as read wrong.

    python training/scripts/evaluate_ocr.py "path/to/Ghanaian License Plates.v2i.yolov11.zip"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_external import unpack  # noqa: E402
from plate_text import decode, normalise  # noqa: E402

OCR_SIZE = (128, 64)  # width, height the reader expects


def crop(im: Image.Image, cx: float, cy: float, w: float, h: float, pad: float) -> Image.Image:
    """Box (normalised centre/size) grown by `pad` of its size on each side, clamped to the image."""
    W, H = im.size
    x0, x1 = (cx - w / 2 - w * pad) * W, (cx + w / 2 + w * pad) * W
    y0, y1 = (cy - h / 2 - h * pad) * H, (cy + h / 2 + h * pad) * H
    return im.crop((max(0, round(x0)), max(0, round(y0)), min(W, round(x1)), min(H, round(y1))))


class Reader:
    def __init__(self, path: Path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def probs(self, plates: list[Image.Image]) -> np.ndarray:
        # Same as the app: stretch to 128x64 (no letterbox), RGB bytes.
        batch = np.stack([np.asarray(p.convert("RGB").resize(OCR_SIZE, Image.BILINEAR), dtype=np.uint8) for p in plates])
        return self.session.run(["plate"], {self.input: batch})[0]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def cer(pred: str, truth: str) -> float:
    """Character error rate: edit distance / length of the truth."""
    d = list(range(len(truth) + 1))
    for i, p in enumerate(pred, 1):
        prev, d[0] = d[0], i
        for j, t in enumerate(truth, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (p != t))
    return d[len(truth)] / max(1, len(truth))


def summarise(rows: list[tuple[str, str]]) -> dict:
    """rows: (prediction, truth) pairs, normalised."""
    by_plate = defaultdict(list)
    for p, t in rows:
        by_plate[t].append(p == t)
    return {
        "plates": len(rows),
        "exact": round(sum(p == t for p, t in rows) / len(rows), 4),
        # Many photos show the same car; this weights each distinct plate equally.
        "exactPerDistinctPlate": round(sum(sum(v) / len(v) for v in by_plate.values()) / len(by_plate), 4),
        "cer": round(sum(cer(p, t) for p, t in rows) / len(rows), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="the Ghanaian License Plates Roboflow export (zip or folder)")
    p.add_argument("--labels", type=Path, default=Path("training/ocr/ghana-plates.csv"))
    p.add_argument("--reader", type=Path, default=Path("web/public/model/plate-reader.fp16.onnx"))
    p.add_argument("--model-dir", type=Path, default=Path("web/public/model"))
    p.add_argument("--pads", type=float, nargs="+", default=[0.1, 0.05, 0.0])
    p.add_argument("--conf", type=float, default=0.35, help="detector threshold (the app's default)")
    p.add_argument("--out", type=Path, default=Path("training/results/ocr-ghana.json"))
    p.add_argument("--errors", type=Path, help="write misread crops here for inspection")
    args = p.parse_args()

    root = unpack(args.dataset, Path("data/external/ghana/raw"))
    files = {f.name: f for f in root.rglob("*.jpg")}
    labels = [r for r in csv.DictReader(open(args.labels, encoding="utf-8")) if r["plate"]]
    reader = Reader(args.reader)
    results: dict = {"labelled": len(labels), "reader": {}, "endToEnd": {}}

    # 1. Reader only, on the labelled boxes, for each crop padding.
    images = {name: Image.open(files[name]).convert("RGB") for name in {r["image"] for r in labels}}
    for pad in args.pads:
        crops = [crop(images[r["image"]], *(float(r[k]) for k in ("cx", "cy", "w", "h")), pad) for r in labels]
        probs = reader.probs(crops)
        for ghana in (False, True):
            reads = [decode(pr.tolist(), ghana=ghana) for pr in probs]
            key = f"pad{pad:g}" + ("+ghana" if ghana else "")
            results["reader"][key] = summarise([(normalise(d["text"]), normalise(r["plate"])) for d, r in zip(reads, labels)])
            print(f"reader   {key:14} {results['reader'][key]}")
            if args.errors and ghana and pad == args.pads[-1]:
                args.errors.mkdir(parents=True, exist_ok=True)
                for i, (d, r, c) in enumerate(zip(reads, labels, crops)):
                    if normalise(d["text"]) != normalise(r["plate"]):
                        c.save(args.errors / f"{i:03d}_{normalise(r['plate'])}_read_{d['text'] or 'nothing'}.png")

    # 2. End to end with each served detector, with the last --pads value (a tight box: the app uses 0).
    from ultralytics import YOLO

    manifest = json.loads((args.model_dir / "models.json").read_text(encoding="utf-8"))
    pad = args.pads[-1]
    for variant, meta in manifest["variants"].items():
        detector = YOLO(str(args.model_dir / meta["file"]), task="detect")
        rows = []
        for name, group in _by_image(labels).items():
            im = images[name]
            W, H = im.size
            found = detector.predict(np.asarray(im)[:, :, ::-1], imgsz=meta["imgsz"], conf=args.conf, verbose=False)[0].boxes.xyxy.tolist()
            for r in group:
                cx, cy, w, h = (float(r[k]) for k in ("cx", "cy", "w", "h"))
                truth = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
                match = max(found, key=lambda b: iou(b, truth), default=None)
                if match is None or iou(match, truth) < 0.5:
                    rows.append(("", normalise(r["plate"])))
                    continue
                x0, y0, x1, y1 = match
                c = crop(im, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H, pad)
                rows.append((normalise(decode(reader.probs([c])[0].tolist())["text"]), normalise(r["plate"])))
        results["endToEnd"][variant] = {**summarise(rows), "detected": round(sum(bool(p) for p, _ in rows) / len(rows), 4)}
        print(f"end2end  {variant:14} {results['endToEnd'][variant]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    record(args.model_dir / "models.json", args.reader, results, f"pad{pad:g}+ghana")


def record(manifest_path: Path, reader: Path, results: dict, key: str) -> None:
    """Puts the scores the app shows into models.json, for each variant that pairs with this reader."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for variant, meta in manifest["variants"].items():
        if meta.get("reader", {}).get("file") != reader.name:
            continue
        meta["reader"]["ghana"] = {
            "plates": results["reader"][key]["plates"],
            "exact": results["reader"][key]["exact"],
            "endToEnd": results["endToEnd"][variant]["exact"],
        }
        print(f"   models.json: {variant} reads {meta['reader']['ghana']}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _by_image(labels):
    out = defaultdict(list)
    for r in labels:
        out[r["image"]].append(r)
    return out


if __name__ == "__main__":
    main()
