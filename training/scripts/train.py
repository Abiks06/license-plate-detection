"""Fine-tune a COCO-pretrained YOLO11 model into a single-class license plate detector.

Two variants ship in the web app:
    fast      YOLO11n  small download, real-time webcam
    accurate  YOLO11s  best accuracy for photos

Defaults target a Colab Pro GPU (A100 / L4). Quick CPU smoke test:
    python training/scripts/train.py --variant fast --device cpu --imgsz 320 --epochs 1 --fraction 0.03 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

VARIANTS = {"fast": "yolo11n.pt", "accurate": "yolo11s.pt"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANTS, default="accurate")
    p.add_argument("--data", type=Path, default=Path("data/yolo/data.yaml"))
    p.add_argument("--imgsz", type=int, default=768, help="plates are small; above 640 helps recall")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30, help="stop after this many epochs without improvement")
    p.add_argument("--batch", type=float, default=-1, help="-1 = auto-fit ~60%% of GPU memory; or a fixed size")
    p.add_argument("--device", default="0", help='"0" for the first GPU, "cpu" otherwise')
    p.add_argument("--fraction", type=float, default=1.0, help="share of the training set to use")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
    model = YOLO(VARIANTS[args.variant])
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=batch,
        device=args.device,
        fraction=args.fraction,
        workers=args.workers,
        # Absolute, or Ultralytics nests a relative project under runs/detect/.
        project=str(Path("runs").resolve()),
        name=args.variant,
        exist_ok=True,
        seed=42,
        deterministic=True,
        cos_lr=True,
        close_mosaic=max(1, min(15, args.epochs // 6)),
        # Plates are flat and text-like: mild geometry only, never upside down.
        fliplr=0.5,
        flipud=0.0,
        degrees=3.0,
        shear=1.0,
        perspective=0.0005,
        plots=True,
    )


if __name__ == "__main__":
    main()
