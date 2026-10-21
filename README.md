# License Plate Detection

A YOLO11 license plate detector that runs **entirely in the browser**: images and webcam frames never leave your device.

> Work in progress: the training pipeline is ready; the PlateSight web app is being built.

## How it works

1. **Data:** [Vehicle Registration Plates](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk) (Augmented Startups, CC BY 4.0), plus Open Images V7 *Vehicle registration plate* for varied street and phone photos.
2. **Training:** COCO-pretrained YOLO11 fine-tuned as a single-class detector, in two sizes:

   | Variant | Model | For |
   | :-- | :-- | :-- |
   | `fast` | YOLO11n | live webcam, small download |
   | `accurate` | YOLO11s | photos, best accuracy |

3. **Export:** ONNX, scored on held-out test images, including an Open Images-only "in the wild" test split so the reported accuracy isn't flattered by near-duplicate CCTV frames.
4. **Inference:** ONNX Runtime Web in a Web Worker, with non-maximum suppression in TypeScript.

## Train it

**On a GPU (recommended):** open [`training/train_colab.ipynb`](training/train_colab.ipynb) in Google Colab, pick a GPU runtime and *Run all*. It downloads `plates-model.zip`; unzip it into the repo root.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abiks06/license-plate-detection/blob/main/training/train_colab.ipynb)

**Locally:**

```bash
python -m venv training/.venv
training/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
training/.venv/Scripts/python -m pip install -r training/requirements.txt
training/.venv/Scripts/python training/scripts/prepare_dataset.py
training/.venv/Scripts/python training/scripts/train.py --variant fast --device cpu --imgsz 416 --epochs 20
training/.venv/Scripts/python training/scripts/export.py --variant fast
```

## Repository layout

```
training/
  scripts/prepare_dataset.py   download + convert datasets to YOLO format
  scripts/train.py             fine-tune YOLO11 (fast / accurate)
  scripts/export.py            evaluate on test splits, export ONNX + models.json
  train_colab.ipynb            one-click GPU training
web/                           PlateSight, the in-browser app (Vite, React, ONNX Runtime Web)
```

## Licenses

Code: [AGPL-3.0](LICENSE), as required by Ultralytics YOLO. Training data: CC BY 4.0 (see above); credit to Augmented Startups / Roboflow Universe and the Open Images authors.
