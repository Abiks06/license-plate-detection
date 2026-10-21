"""Quantise an exported detector to INT8 and keep it only if accuracy holds.

Static QDQ quantisation, calibrated on real training images, typically makes the
ONNX file ~4x smaller and faster on CPU / WebAssembly. The detection head (the
last layers that decode boxes) stays in float, where quantisation error hurts most.

Both models are scored on the test split; models.json switches the variant to the
INT8 file only if mAP@50 drops by no more than --max-drop. Run after export.py:
    python training/scripts/optimize_model.py --variant fast
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from ultralytics import YOLO


def letterbox(path: Path, size: int) -> np.ndarray:
    """Same preprocessing as the web worker: fit, pad with grey 114, CHW float in [0, 1]."""
    im = Image.open(path).convert("RGB")
    scale = min(size / im.width, size / im.height)
    w, h = round(im.width * scale), round(im.height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((w, h), Image.BILINEAR), ((size - w) // 2, (size - h) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


class Calibration(CalibrationDataReader):
    def __init__(self, images: list[Path], size: int, input_name: str):
        self.batches = iter([{input_name: letterbox(p, size)} for p in images])

    def get_next(self):
        return next(self.batches, None)


def head_nodes(model: onnx.ModelProto) -> list[str]:
    """Box-decoding nodes of the final layer (DFL, sigmoid, concat, arithmetic) to keep in float.

    Node names look like '/model.23/dfl/conv/Conv'; the highest index is the Detect head.
    Its two conv branches (cv2: boxes, cv3: classes) still get quantised: they're most
    of its compute and tolerate it well. Everything after them stays float.
    """
    def layer(n):
        part = n.name.split("/")[1] if n.name.startswith("/model.") else ""
        return int(part.split(".")[1]) if part.startswith("model.") and part.split(".")[1].isdigit() else -1

    last = max(layer(n) for n in model.graph.node)
    return [n.name for n in model.graph.node if layer(n) == last and "/cv2." not in n.name and "/cv3." not in n.name]


def latency(path: Path, size: int, runs: int = 20) -> float:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, size, size).astype(np.float32)
    name = s.get_inputs()[0].name
    for _ in range(3):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(runs):
        s.run(None, {name: x})
    return (time.perf_counter() - t) / runs * 1000


def score(path: Path, data: Path, size: int) -> float:
    m = YOLO(str(path), task="detect").val(data=str(data), split="test", imgsz=size, device="cpu", plots=False, verbose=False)
    return round(float(m.box.map50), 4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["fast", "accurate"], required=True)
    p.add_argument("--model-dir", type=Path, default=Path("web/public/model"))
    p.add_argument("--data", type=Path, default=Path("data/yolo"))
    p.add_argument("--calibration", type=int, default=200, help="training images used to calibrate")
    p.add_argument("--max-drop", type=float, default=0.01, help="largest acceptable mAP@50 loss")
    args = p.parse_args()

    manifest_path = args.model_dir / "models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = manifest["variants"][args.variant]
    size = meta["imgsz"]
    fp32 = args.model_dir / f"plates-{args.variant}.onnx"
    int8 = args.model_dir / f"plates-{args.variant}.int8.onnx"
    prepped = args.model_dir / f"plates-{args.variant}.prep.onnx"

    random.seed(0)
    calib = random.sample(sorted((args.data / "images" / "train").glob("*.jpg")), args.calibration)
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
    base = onnx.load(str(prepped))
    print(f"[{args.variant}] calibrating INT8 on {len(calib)} images")
    quantize_static(
        str(prepped),
        str(int8),
        Calibration(calib, size, base.graph.input[0].name),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=head_nodes(base),
    )
    prepped.unlink()

    print(f"[{args.variant}] scoring float vs INT8 on the test split")
    f_map, q_map = score(fp32, args.data / "data.yaml", size), score(int8, args.data / "data.yaml", size)
    f_ms, q_ms = latency(fp32, size), latency(int8, size)
    keep = f_map - q_map <= args.max_drop
    result = {
        "file": int8.name,
        "bytes": int8.stat().st_size,
        "mAP50": q_map,
        "floatMAP50": f_map,
        "cpuMsFloat": round(f_ms, 1),
        "cpuMsInt8": round(q_ms, 1),
        "adopted": keep,
    }
    print("  ", json.dumps(result))

    meta["int8"] = result
    meta["floatFile"], meta["floatBytes"] = fp32.name, fp32.stat().st_size
    if keep:
        meta["file"], meta["bytes"] = int8.name, int8.stat().st_size
    else:
        meta["file"], meta["bytes"] = fp32.name, fp32.stat().st_size
        int8.unlink()
        print(f"   INT8 lost {f_map - q_map:.3f} mAP@50 (> {args.max_drop}); keeping the float model")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"   models.json now serves {meta['file']}")


if __name__ == "__main__":
    main()
