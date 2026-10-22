"""Measure accuracy vs speed of a trained variant at several inference sizes.

Browser cost grows with the square of the input size, and a YOLO model trained at
768px still works at smaller sizes. This prints mAP on the test split(s) and ONNX
Runtime CPU latency per size, to choose the size to export:
    python training/scripts/benchmark_sizes.py --variant fast --sizes 512 640 768
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def latency(onnx_path: str, size: int, runs: int = 15) -> float:
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, size, size).astype(np.float32)
    name = s.get_inputs()[0].name
    for _ in range(3):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(runs):
        s.run(None, {name: x})
    return (time.perf_counter() - t) / runs * 1000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["fast", "accurate"], required=True)
    p.add_argument("--weights", type=Path, help="defaults to training/results/<variant>/best.pt")
    p.add_argument("--data", type=Path, default=Path("data/yolo"))
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 640, 768])
    args = p.parse_args()

    weights = args.weights or Path(f"training/results/{args.variant}/best.pt")
    sets = {"all": args.data / "data.yaml"}
    if (args.data / "data-oi.yaml").exists():
        sets["inTheWild"] = args.data / "data-oi.yaml"

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for size in args.sizes:
            model = YOLO(str(weights))
            row = {"imgsz": size}
            for name, data in sets.items():
                m = model.val(data=str(data), split="test", imgsz=size, device="cpu", plots=False, verbose=False, project=tmp)
                row[name] = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4), "recall": round(float(m.box.mr), 4)}
            onnx_path = YOLO(str(weights)).export(format="onnx", imgsz=size, opset=17, simplify=True, dynamic=False)
            row["cpuMs"] = round(latency(onnx_path, size), 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("\nimgsz | " + " | ".join(f"{k} mAP50" for k in sets) + " | CPU ms")
    for r in rows:
        print(f"{r['imgsz']:5d} | " + " | ".join(f"{r[k]['mAP50']:.3f}" for k in sets) + f" | {r['cpuMs']}")


if __name__ == "__main__":
    main()
