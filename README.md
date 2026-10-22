# PlateSight: License Plate Detection

A YOLO11 license plate detector that runs **entirely in your browser**. Detect plates in photos or live from your webcam, then redact them in one click. Images and frames never leave your device.

**[Live demo → platesight-xi.vercel.app](https://platesight-xi.vercel.app)**

![PlateSight detecting a license plate](docs/screenshot.jpg)

> This is a rebuild of the license plate detection model from my Research Methods group project at Academic City University (Feb–May 2026), which I led. The training pipeline, models and web app in this repository were built from scratch in October 2026.

## Features

- **On-device inference:** ONNX Runtime Web in a Web Worker on multi-threaded WebAssembly (cross-origin isolated). Nothing is uploaded.
- **Two models:** *Fast* (YOLO11n, 640px) for live video and *Accurate* (YOLO11s, 768px) for photos, with each one's measured test accuracy shown in the app.
- **Image and live camera modes:** drag and drop, paste, browse, eight test-set samples, or real-time webcam detection with an FPS meter.
- **One-click redaction:** plates are pixelated (not blurred, so they can't be sharpened back) and the image downloads locally.
- **Hardened delivery:** strict Content-Security-Policy, self-hosted fonts, immutable caching for hashed assets, and the runtime and first model preloaded from the HTML.

## Results

Both models were trained on a Colab GPU for up to 120 epochs at 768px with early stopping (Fast peaked at epoch 76, Accurate at 64). Scores below are for the **exact ONNX files the app serves**, evaluated at the input size the app uses.

| Model | Input | mAP@50 in the wild | mAP@50–95 in the wild | Precision / Recall (wild) | mAP@50, full test set | Download |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Fast** · YOLO11n | 640px | **0.873** | 0.583 | 0.945 / 0.821 | 0.907 | **5.4 MB** |
| **Accurate** · YOLO11s | 768px | **0.894** | 0.588 | 0.930 / 0.838 | 0.922 | **19.1 MB** |

- **In the wild:** 2,065 held-out Open Images photos (2,843 plates) of streets, cars and phone shots. This is the honest number.
- **Full test set:** those photos plus 882 held-out Roboflow images. The Roboflow split is mostly parking-garage CCTV with near-duplicate frames, which flatters scores, so it isn't reported alone.
- On the eight varied sample images in the app, both models find all 8 real plates with no false positives (74–88% confidence). The dataset also labels a "CityBus" sign as a plate, which both models correctly ignore.

### Optimisation decisions

Every compression option goes through the same gate: it ships only if mAP@50 drops by at most 1 point on **both** test sets.

| Option | Fast | Accurate | Outcome |
| :-- | :-- | :-- | :-- |
| Input size 768 → 640px | −1.0 point in the wild, 1.5× faster | kept at 768 | Fast's job is live video, so speed wins there |
| INT8 static quantisation | −10.0 points in the wild | −1.9 points | **Rejected** by the gate; Accurate INT8 would score below Fast |
| Float16 weight storage | 10.7 → 5.4 MB, ±0.000 | 38.0 → 19.1 MB, ±0.0001 | **Adopted:** weights are stored in half precision and computed in float32 |

### Speed

Measured in Edge on an Intel i7-1255U laptop **running on battery** (Balanced power plan), 4 WebAssembly threads:

| Model | Median per image |
| :-- | :-- |
| Fast | ~400 ms |
| Accurate | ~1.35 s |

Under the same battery conditions, the earlier 320px CPU stand-in took ~70 ms, versus ~26 ms on mains power. That suggests roughly 2.5–3× faster times on a plugged-in laptop: about 130 ms (Fast) and 0.5 s (Accurate).

### Limitations

- Training data is mostly Vietnamese CCTV and Open Images photos; accuracy on Ghanaian plates hasn't been measured.
- Detection only: plates are located, not read (no OCR).
- WebAssembly only. A WebGPU backend would speed up the Accurate model, at the cost of a larger runtime download.

## How it works

1. **Data:** [Vehicle Registration Plates](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk) (Augmented Startups, CC BY 4.0), plus up to 6,000 Open Images V7 *Vehicle registration plate* training photos for variety.
2. **Training:** a COCO-pretrained YOLO11 convolutional neural network fine-tuned as a single-class detector.
3. **Size and precision:** `benchmark_sizes.py` measures accuracy against speed per input size; `optimize_model.py` tries INT8, then float16 weights, behind the accuracy gate.
4. **Inference:** letterbox to the model's input size on an OffscreenCanvas, run the model, decode boxes and apply non-maximum suppression in TypeScript, then map back to the original pixels.

## Train it

**On a GPU (recommended):** open the notebook in Colab, pick a GPU runtime and *Run all*. It downloads `plates-model.zip`; unzip it into the repo root.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abiks06/license-plate-detection/blob/main/training/train_colab.ipynb)

Then, locally, choose the input size and optimise:

```bash
training/.venv/Scripts/python training/scripts/prepare_dataset.py --open-images 1 --open-images-splits test
training/.venv/Scripts/python training/scripts/benchmark_sizes.py --variant fast --sizes 512 640 768
training/.venv/Scripts/python training/scripts/export.py --variant fast --weights training/results/fast/best.pt --imgsz 640
training/.venv/Scripts/python training/scripts/optimize_model.py --variant fast
training/.venv/Scripts/python training/scripts/optimize_model.py --variant accurate
```

**Quick CPU-only run:**

```bash
python -m venv training/.venv
training/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
training/.venv/Scripts/python -m pip install -r training/requirements.txt
training/.venv/Scripts/python training/scripts/prepare_dataset.py
training/.venv/Scripts/python training/scripts/train.py --variant fast --device cpu --imgsz 320 --epochs 3 --fraction 0.15
training/.venv/Scripts/python training/scripts/export.py --variant fast
```

**Run the app:** see [`web/README.md`](web/README.md).

## Repository layout

```
training/
  scripts/prepare_dataset.py   download + convert datasets to YOLO format
  scripts/train.py             fine-tune YOLO11 (fast / accurate)
  scripts/benchmark_sizes.py   accuracy vs speed per input size
  scripts/export.py            evaluate on test splits, export ONNX + models.json
  scripts/optimize_model.py    INT8 / float16 candidates behind an accuracy gate
  results/{fast,accurate}/     trained weights, training curves, PR curves, confusion matrices
  train_colab.ipynb            one-click GPU training
web/                           PlateSight (Vite, React, Tailwind, Motion, ONNX Runtime Web)
docs/                          README assets
```

## Licenses

Code and models: [AGPL-3.0](LICENSE), as required by Ultralytics YOLO. Training data: CC BY 4.0; credit to Augmented Startups / Roboflow Universe and the Open Images authors. Sample images in the app come from the dataset's test split.
