# PlateSight: License Plate Detection

A YOLO11 license plate detector that runs **entirely in your browser**. Detect plates in photos or live from your webcam, then redact them in one click. Images and frames never leave your device.

**[Live demo → platesight-xi.vercel.app](https://platesight-xi.vercel.app)**

![PlateSight detecting a license plate](docs/screenshot.jpg)

## Features

- **On-device inference:** ONNX Runtime Web in a Web Worker, multi-threaded WebAssembly (cross-origin isolated), about 30–40 ms per image on a laptop.
- **Image and live camera modes:** drag and drop, paste, browse, eight test-set samples, or real-time webcam detection with an FPS meter.
- **One-click redaction:** plates are pixelated (not blurred, so they can't be sharpened back) and the image downloads locally.
- **Two models:** *Fast* (YOLO11n) for live video and *Accurate* (YOLO11s) for photos, each with its measured test accuracy shown in the app.
- **Hardened delivery:** strict Content-Security-Policy, self-hosted fonts, immutable caching for hashed assets, and the runtime and model preloaded from the HTML.

## Results

> The deployed model is currently a **CPU-trained stand-in** (YOLO11n, 3 epochs at 320px on 15% of the data, 18 minutes on a laptop). The GPU-trained models from [`train_colab.ipynb`](training/train_colab.ipynb) replace it; this table will be updated then.

| Model | Test mAP@50 | Precision | Recall | Size | Browser latency |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Stand-in, float32 | 0.911 | 0.919 | 0.845 | 10.5 MB | – |
| Stand-in, **INT8** (served) | 0.908 | – | – | **3.1 MB** | ~30–40 ms |

The Roboflow test split is mostly parking-garage CCTV with near-duplicate frames, which flatters these numbers. On eight hand-picked varied test images, the stand-in found 7 of 9 labelled plates with one false positive and missed a tilted red plate. The GPU run adds Open Images photos and reports a separate **in-the-wild** score for exactly this reason.

## How it works

1. **Data:** [Vehicle Registration Plates](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk) (Augmented Startups, CC BY 4.0), plus Open Images V7 *Vehicle registration plate* for varied street and phone photos.
2. **Training:** a COCO-pretrained YOLO11 convolutional neural network fine-tuned as a single-class detector.
3. **Export and quantisation:** ONNX, then static INT8 (QDQ) quantisation calibrated on training images, with the box-decoding layers kept in float. INT8 is used only if it costs at most 1 point of mAP@50.
4. **Inference:** letterbox to the model's input size on an OffscreenCanvas, run the model, decode boxes and apply non-maximum suppression in TypeScript, then map back to the original pixels.

## Train it

**On a GPU (recommended):** open the notebook in Colab, pick a GPU runtime and *Run all*. It downloads `plates-model.zip`; unzip it into the repo root, then run the optimiser.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abiks06/license-plate-detection/blob/main/training/train_colab.ipynb)

**Locally:**

```bash
python -m venv training/.venv
training/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
training/.venv/Scripts/python -m pip install -r training/requirements.txt
training/.venv/Scripts/python training/scripts/prepare_dataset.py
training/.venv/Scripts/python training/scripts/train.py --variant fast --device cpu --imgsz 320 --epochs 3 --fraction 0.15
training/.venv/Scripts/python training/scripts/export.py --variant fast
training/.venv/Scripts/python training/scripts/optimize_model.py --variant fast
```

**Run the app:** see [`web/README.md`](web/README.md).

## Repository layout

```
training/
  scripts/prepare_dataset.py   download + convert datasets to YOLO format
  scripts/train.py             fine-tune YOLO11 (fast / accurate)
  scripts/export.py            evaluate on test splits, export ONNX + models.json
  scripts/optimize_model.py    INT8 quantisation behind an accuracy gate
  train_colab.ipynb            one-click GPU training
web/                           PlateSight (Vite, React, Tailwind, Motion, ONNX Runtime Web)
docs/                          README assets
```

## Licenses

Code: [AGPL-3.0](LICENSE), as required by Ultralytics YOLO. Training data: CC BY 4.0; credit to Augmented Startups / Roboflow Universe and the Open Images authors. Sample images in the app come from the dataset's test split.
