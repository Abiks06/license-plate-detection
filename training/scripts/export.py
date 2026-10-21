"""Evaluate a trained variant on the held-out test split(s) and export it for the browser.

Writes into --out (default web/public/model):
    plates-<variant>.onnx   fixed input [1, 3, imgsz, imgsz], raw YOLO output [1, 5, anchors]
    models.json             per-variant input size, test metrics and file size (merged, so
                            exporting "fast" then "accurate" leaves both entries)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

DATASETS = [
    {"name": "Vehicle Registration Plates (Augmented Startups, Roboflow Universe)", "license": "CC BY 4.0",
     "url": "https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk"},
    {"name": "Open Images V7: Vehicle registration plate", "license": "CC BY 4.0 annotations; images CC BY 2.0",
     "url": "https://storage.googleapis.com/openimages/web/index.html"},
]


def evaluate(model: YOLO, data: Path, imgsz: int, device: str) -> dict:
    m = model.val(data=str(data), split="test", imgsz=imgsz, device=device, plots=False, verbose=False)
    return {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["fast", "accurate"], required=True)
    p.add_argument("--weights", type=Path, help="defaults to runs/<variant>/weights/best.pt")
    p.add_argument("--data", type=Path, default=Path("data/yolo"))
    p.add_argument("--imgsz", type=int, default=None, help="defaults to the training size")
    p.add_argument("--out", type=Path, default=Path("web/public/model"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    weights = args.weights or Path(f"runs/{args.variant}/weights/best.pt")
    if not weights.exists():
        # Older runs may sit under runs/detect/... depending on the Ultralytics version.
        found = sorted(Path("runs").rglob(f"{args.variant}/weights/best.pt"))
        if not found:
            raise SystemExit(f"No weights for '{args.variant}': train it first or pass --weights")
        weights = found[-1]
    model = YOLO(str(weights))
    train_args = model.ckpt.get("train_args", {})
    imgsz = args.imgsz or int(train_args.get("imgsz", 640))

    print(f"[{args.variant}] evaluating at {imgsz}px")
    metrics = {"all": evaluate(model, args.data / "data.yaml", imgsz, args.device)}
    if (args.data / "data-oi.yaml").exists():
        metrics["inTheWild"] = evaluate(model, args.data / "data-oi.yaml", imgsz, args.device)
    print("  ", metrics)

    print(f"[{args.variant}] exporting ONNX")
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False, half=False))

    args.out.mkdir(parents=True, exist_ok=True)
    filename = f"plates-{args.variant}.onnx"
    shutil.copy2(onnx_path, args.out / filename)

    manifest_path = args.out / "models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("variants", {})[args.variant] = {
        "file": filename,
        "bytes": (args.out / filename).stat().st_size,
        "architecture": "YOLO11n" if args.variant == "fast" else "YOLO11s",
        "imgsz": imgsz,
        "classes": ["license_plate"],
        "metrics": metrics,
        "training": {
            "epochs": train_args.get("epochs"),
            "imgsz": train_args.get("imgsz"),
            "fraction": train_args.get("fraction", 1.0),
            "device": str(train_args.get("device")),
            "openImages": (args.data / "data-oi.yaml").exists(),
        },
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest["output"] = "raw [1, 4 + classes, anchors]: cx, cy, w, h in input pixels, then class scores"
    manifest["datasets"] = DATASETS
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out / filename} and models.json")


if __name__ == "__main__":
    main()
