# PlateSight: License Plate Recognition

Reads license plate numbers **entirely in your browser**, tuned for Ghanaian plates. A YOLO11 detector finds each plate, a plate reader turns it into text (`GR 3966-21`), and Ghana's plate formats correct the reading. Works on photos or live from your webcam, with one-click redaction. Images and frames never leave your device.

**[Live demo → platesight-xi.vercel.app](https://platesight-xi.vercel.app)**

![PlateSight reading a Ghanaian plate: GG 3096-17](docs/screenshot.jpg)

> This is a rebuild of the license plate detection model from my Research Methods group project at Academic City University (Feb–May 2026), which I led. The training pipeline, models and web app in this repository were built from scratch in October 2026.

## Features

- **On-device inference:** ONNX Runtime Web in Web Workers. It answers on the CPU (multi-threaded WebAssembly) within about two seconds, and on devices with a hardware GPU it warms up WebGPU in the background and switches over when ready. Nothing is uploaded.
- **Plate numbers, not just boxes:** each detected plate is read by a compact vision transformer and decoded against Ghana's plate formats, shown as printed (`GR 3966-21`, `GT 9556 V`) with a copy button and uncertain characters underlined. Foreign plates are read as they are.
- **Two models:** *Fast* (YOLO11n at 640px + the XS reader) for live video and *Accurate* (YOLO11s at 768px + the S reader) for photos, with each one's measured accuracy shown in the app.
- **Image and live camera modes:** drag and drop, paste, browse, twelve samples (four Ghanaian), or real-time webcam reading with an FPS meter.
- **One-click redaction:** plates are pixelated (not blurred, so they can't be sharpened back) and the image downloads locally.
- **Hardened delivery:** strict Content-Security-Policy, self-hosted fonts, immutable caching for hashed assets, and the first model preloaded from the HTML.

## Results

### Reading the plate number

Measured on **310 hand-labelled Ghanaian plates** (111 distinct) from the public-domain [Ghanaian License Plates](https://universe.roboflow.com/luther-eyandi/ghanaian-license-plates) dataset, which no model here trained on. I transcribed all 380 plates in it blind (without seeing any model's reading); 70 too blurred or cut off for a person to read are excluded. Labels: `training/ocr/ghana-plates.csv`. "Exact" means every character right.

| | Reader | Download | Exact, reader on the labelled box | **Exact, end to end** | Characters wrong |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Fast** | CCT-XS | 1.8 MB | 81.0% | **76.8%** | 4.6% |
| **Accurate** | CCT-S | 2.8 MB | 83.5% | **79.4%** | 3.2% |

- **End to end** is what the app does: the detector finds the plate at the default 35% threshold (it finds 94%), then the reader reads its box. A missed plate counts as wrong.
- **Ghana formats help:** the reader predicts each of its 10 character slots independently, so it can put a letter in the year (`AS 4633-I2`). Choosing the most probable reading that fits a Ghanaian pattern (two letters, 1–4 digits, then a two-digit year or one letter) lifts the S reader from 79.0% to 83.5% on labelled boxes. If the free reading is far more probable than any Ghanaian one (a foreign plate), it's kept instead.
- **What still fails:** Ghana's slashed zero, which the reader sometimes takes for a `D` or drops (`GR 3842-09` → `GR3842D`); motion blur; and steep angles. The readers are [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)'s global models, trained on plates from 60+ countries but not Ghana, so fine-tuning on Ghanaian plates is the clear next step.
- **Choices, and how they were made:** the crop margin (0% for S, 5% for XS) was picked from three options on this same set, so treat these as slightly optimistic; there's no separate validation set yet. Averaging three crops and a prior against letter-ending plates each moved results by under a point (noise at this size) and were left out.
- **Speed:** the readers run on the CPU (WebAssembly) even when the detector uses WebGPU: at this size GPU transfer overhead dominated (~100 ms per plate on WebGPU vs 13–50 ms for XS on the CPU, on battery). Float16 weights halve both downloads with identical readings on all 310 plates.

Reproduce with `training/scripts/evaluate_ocr.py` (scores in `training/results/ocr-ghana*.json`).

### Finding the plate

Both models were trained on a Colab GPU for up to 120 epochs at 768px with early stopping (Fast peaked at epoch 76, Accurate at 64). Scores below are for the **exact ONNX files the app serves**, evaluated at the input size the app uses.

| Model | Input | mAP@50 in the wild | mAP@50–95 in the wild | Precision / Recall (wild) | mAP@50, Ghanaian plates | mAP@50, full test set | Download |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Fast** · YOLO11n | 640px | **0.873** | 0.583 | 0.945 / 0.821 | **0.956** | 0.907 | **5.4 MB** |
| **Accurate** · YOLO11s | 768px | **0.894** | 0.588 | 0.930 / 0.838 | **0.954** | 0.922 | **19.1 MB** |

- **In the wild:** 2,065 held-out Open Images photos (2,843 plates) of streets, cars and phone shots. This is the honest number.
- **Full test set:** those photos plus 882 held-out Roboflow images. The Roboflow split is mostly parking-garage CCTV with near-duplicate frames, which flatters scores, so it isn't reported alone.
- **Ghanaian plates:** all 321 photos (380 plates) of [Ghanaian License Plates](https://universe.roboflow.com/luther-eyandi/ghanaian-license-plates) (public domain), a dataset the models never saw: phone shots of cars in Accra car parks plus a few plate graphics. Neither model was trained on a single Ghanaian plate. Precision / recall: Fast 0.948 / 0.904, Accurate 0.918 / 0.918. Looking through the errors, many "false positives" are real plates on background cars that the dataset didn't label, so precision is understated; the genuine misses are mostly motion-blurred close-ups. The photos were stretched to 640×640 by the dataset's export, which a real camera wouldn't do. Scores come from `training/results/ghana.json`.
- On the eight varied sample images in the app, both models find all 8 real plates with no false positives (74–88% confidence). The dataset also labels a "CityBus" sign as a plate, which both models correctly ignore.

### Optimisation decisions

Every compression option goes through the same gate: it ships only if mAP@50 drops by at most 1 point on **both** test sets.

| Option | Fast | Accurate | Outcome |
| :-- | :-- | :-- | :-- |
| Input size 768 → 640px | −1.0 point in the wild, 1.5× faster | kept at 768 | Fast's job is live video, so speed wins there |
| INT8 static quantisation | −10.0 points in the wild | −1.9 points | **Rejected** by the gate; Accurate INT8 would score below Fast |
| Float16 weight storage | 10.7 → 5.4 MB, ±0.000 | 38.0 → 19.1 MB, ±0.0001 | **Adopted:** weights are stored in half precision and computed in float32 |

### Speed

Median per image on the eight samples, in Edge on an Intel i7-1255U laptop with Iris Xe graphics, **running on battery** (Balanced power plan). Both backends find the same plates.

| Model | WebAssembly (CPU, 4 threads) | WebGPU (Iris Xe) | Speed-up |
| :-- | :-- | :-- | :-- |
| Fast | 382 ms | **179 ms** | 2.1× |
| Accurate | 1,432 ms | **269 ms** | 5.3× |

WebGPU's first run compiles GPU shaders: about 8 s on a first visit, about 0.3 s afterwards because the browser caches them. So the app never waits for it: it is ready on the CPU as soon as the model has arrived, warms WebGPU in a second worker, and hands over once that worker is ready for the selected model. The WebGPU runtime is only fetched after the CPU worker is ready, so it never slows the first detection. On a repeat visit (files cached) the live site is ready in about 2 s and on WebGPU by about 6 s; a first visit depends on how fast the 5.4 MB model downloads. Devices without a hardware GPU never download the WebGPU runtime. Append `?backend=wasm` or `?backend=webgpu` to the URL to force one backend.

On mains power the CPU path is roughly 2.5–3× faster than these battery figures.

### Limitations

- Training data is mostly Vietnamese CCTV and Open Images photos. It transfers well to Ghanaian plates (0.95 mAP@50), but that test set is 321 car-park photos from one dataset; night scenes, moving traffic and motorbike plates are untested.
- The plate readers have never seen a Ghanaian plate; about 1 in 5 Ghanaian plates is misread somewhere. Underlined characters in the app are the ones to double-check.
- On GPU devices the first visit downloads both runtimes (about 3.5 MB + 6.4 MB compressed); both are cached after that.

## How it works

1. **Data:** [Vehicle Registration Plates](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk) (Augmented Startups, CC BY 4.0), plus up to 6,000 Open Images V7 *Vehicle registration plate* training photos for variety.
2. **Training:** a COCO-pretrained YOLO11 convolutional neural network fine-tuned as a single-class detector.
3. **Size and precision:** `benchmark_sizes.py` measures accuracy against speed per input size; `optimize_model.py` tries INT8, then float16 weights, behind the accuracy gate.
4. **Finding:** letterbox to the detector's input size on an OffscreenCanvas, run it, decode boxes and apply non-maximum suppression in TypeScript, then map back to the original pixels.
5. **Reading:** each box is stretched to the reader's 128×64 input (as it was trained, no letterbox) and all plates are read in one batch. `web/src/lib/plateText.ts` (mirrored by `training/scripts/plate_text.py`, which the scores come from) turns the per-slot probabilities into the most probable Ghanaian-format reading.

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

### Test on Ghanaian plates

`evaluate_external.py` scores the served ONNX files on any dataset they never trained on. Download the Roboflow Universe datasets [Ghanaian License Plates](https://universe.roboflow.com/luther-eyandi/ghanaian-license-plates) (public domain) and [Ghana License Plate Detector](https://universe.roboflow.com/aklikaisaac300-gmail-com/ghana-license-plate-detector) (CC BY 4.0) in *YOLOv11* format, then:

```bash
training/.venv/Scripts/python training/scripts/evaluate_external.py --name ghana path/to/first.zip path/to/second.zip
```

It merges every split into one test set, keeps only plate classes, removes photos shared between the datasets and Roboflow's augmented copies, and writes `training/results/ghana.json` (plus an unaugmented "clean" subset score when an export was augmented). It also adds the scores and dataset credit to `models.json`, which the app shows. Pass several zips to merge datasets; the results above use the first dataset only.

To score plate reading on the same photos (uses the transcriptions in `training/ocr/ghana-plates.csv`):

```bash
training/.venv/Scripts/python -m pip install "fast-plate-ocr[onnx]"   # only to download the reader models
training/.venv/Scripts/python training/scripts/evaluate_ocr.py "path/to/Ghanaian License Plates.v2i.yolov11.zip" --reader web/public/model/plate-reader.fp16.onnx
training/.venv/Scripts/python training/scripts/evaluate_ocr.py "path/to/Ghanaian License Plates.v2i.yolov11.zip" --reader web/public/model/plate-reader-xs.fp16.onnx --pads 0 0.1 0.05 --out training/results/ocr-ghana-xs.json
```

The last `--pads` value is the crop margin used end to end; the scores go into `models.json` for the variant that pairs with that reader.

## Repository layout

```
training/
  scripts/prepare_dataset.py   download + convert datasets to YOLO format
  scripts/train.py             fine-tune YOLO11 (fast / accurate)
  scripts/benchmark_sizes.py   accuracy vs speed per input size
  scripts/export.py            evaluate on test splits, export ONNX + models.json
  scripts/optimize_model.py    INT8 / float16 candidates behind an accuracy gate
  scripts/evaluate_external.py score the served models on an outside dataset (e.g. Ghana)
  scripts/evaluate_ocr.py      how often plate numbers are read exactly, reader-only and end to end
  scripts/plate_text.py        Ghana-format decoding of the reader's output
  scripts/add_ghana_samples.py Ghanaian sample photos for the app
  ocr/ghana-plates.csv         hand-transcribed plate numbers for the Ghana dataset
  results/{fast,accurate}/     trained weights, training curves, PR curves, confusion matrices
  train_colab.ipynb            one-click GPU training
web/                           PlateSight (Vite, React, Tailwind, Motion, ONNX Runtime Web)
docs/                          README assets
```

## Licenses

Code and detectors: [AGPL-3.0](LICENSE), as required by Ultralytics YOLO. Plate readers: [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) by ankandrew, MIT. Training data: CC BY 4.0; credit to Augmented Startups / Roboflow Universe and the Open Images authors. Sample images come from that dataset's test split and from the public-domain Ghanaian License Plates dataset (luther-eyandi, Roboflow Universe), which is also the source of the Ghana evaluation.
