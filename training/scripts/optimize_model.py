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


def score(path: Path, data: Path, size: int) -> dict:
    m = YOLO(str(path), task="detect").val(data=str(data), split="test", imgsz=size, device="cpu", plots=False, verbose=False)
    return {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }


def score_all(path: Path, data_dir: Path, size: int) -> dict:
    """Metrics on the full test split and, when available, the Open Images "in the wild" split."""
    out = {"all": score(path, data_dir / "data.yaml", size)}
    if (data_dir / "data-oi.yaml").exists():
        out["inTheWild"] = score(path, data_dir / "data-oi.yaml", size)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["fast", "accurate"], required=True)
    p.add_argument("--model-dir", type=Path, default=Path("web/public/model"))
    p.add_argument("--data", type=Path, default=Path("data/yolo"))
    p.add_argument("--calibration", type=int, default=200, help="training images used to calibrate")
    p.add_argument("--max-drop", type=float, default=0.01, help="largest acceptable mAP@50 loss")
    p.add_argument("--skip-int8", action="store_true", help="go straight to the float16-weights candidate")
    args = p.parse_args()

    manifest_path = args.model_dir / "models.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = manifest["variants"][args.variant]
    size = meta["imgsz"]
    fp32 = args.model_dir / f"plates-{args.variant}.onnx"

    print(f"[{args.variant}] scoring the float model")
    f_metrics = score_all(fp32, args.data, size)
    f_ms = latency(fp32, size)
    meta["floatFile"], meta["floatBytes"], meta["floatMetrics"] = fp32.name, fp32.stat().st_size, f_metrics
    meta["file"], meta["bytes"], meta["metrics"] = fp32.name, fp32.stat().st_size, f_metrics
    meta["candidates"] = {}

    # Smallest first; serve the first candidate whose mAP@50 holds on every test set.
    candidates = [] if args.skip_int8 else [("int8", lambda out: make_int8(fp32, out, args, size))]
    candidates.append(("fp16", lambda out: make_fp16_weights(fp32, out)))
    for name, build in candidates:
        out = args.model_dir / f"plates-{args.variant}.{name}.onnx"
        print(f"[{args.variant}] building {name} candidate")
        build(out)
        c_metrics = score_all(out, args.data, size)
        drops = {k: round(f_metrics[k]["mAP50"] - c_metrics[k]["mAP50"], 4) for k in f_metrics}
        keep = all(d <= args.max_drop for d in drops.values())
        result = {
            "bytes": out.stat().st_size,
            "metrics": c_metrics,
            "mAP50Drop": drops,
            "cpuMs": round(latency(out, size), 1),
            "cpuMsFloat": round(f_ms, 1),
            "adopted": keep,
        }
        print("  ", json.dumps(result))
        meta["candidates"][name] = result
        if keep:
            # The app shows the metrics of the file it serves.
            meta["file"], meta["bytes"], meta["metrics"] = out.name, out.stat().st_size, c_metrics
            break
        out.unlink()
        print(f"   {name} lost {drops} mAP@50 (limit {args.max_drop}); trying the next option")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"   models.json now serves {meta['file']}")


def make_int8(fp32: Path, out: Path, args: argparse.Namespace, size: int) -> None:
    """Static QDQ INT8, calibrated on training images, decode head kept in float."""
    prepped = out.with_suffix(".prep.onnx")
    random.seed(0)
    calib = random.sample(sorted((args.data / "images" / "train").glob("*.jpg")), args.calibration)
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
    base = onnx.load(str(prepped))
    quantize_static(
        str(prepped),
        str(out),
        Calibration(calib, size, base.graph.input[0].name),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=head_nodes(base),
    )
    prepped.unlink()


def make_fp16_weights(fp32: Path, out: Path) -> None:
    """Stores float weights as float16 (half the download) and casts them back to float32.

    ONNX Runtime constant-folds those Casts when the session is created, so inference
    still runs entirely in float32: only the weights' precision changes, not the maths.
    """
    model = onnx.load(str(fp32))
    graph = model.graph
    casts = []
    for init in list(graph.initializer):
        if init.data_type != onnx.TensorProto.FLOAT or np.prod(init.dims) < 16:
            continue
        weights = onnx.numpy_helper.to_array(init).astype(np.float16)
        half = onnx.numpy_helper.from_array(weights, name=f"{init.name}__fp16")
        graph.initializer.remove(init)
        graph.initializer.append(half)
        casts.append(onnx.helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT, name=f"{init.name}__cast"))
    # Casts first, so every consumer still sees its original float32 tensor name.
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.checker.check_model(model)
    onnx.save(model, str(out))


if __name__ == "__main__":
    main()
