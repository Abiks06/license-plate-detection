"""Download the training data and convert it to YOLO format.

Sources:
  * Roboflow Universe "Vehicle Registration Plates" (Augmented Startups, CC BY 4.0),
    mirrored on Hugging Face as keremberke/license-plate-object-detection. Mostly
    parking-garage CCTV frames, so on its own a model trained on it generalises poorly.
  * Optional (--open-images N): Open Images V7 "Vehicle registration plate"
    (annotations CC BY 4.0, images mostly CC BY 2.0) for varied street and phone photos.
    Needs `pip install fiftyone`; best run on Colab.

Output layout (under <data>/yolo):
    images/{train,val,test}/*.jpg   Open Images files are prefixed "oi_"
    labels/{train,val,test}/*.txt   one "0 cx cy w h" line per plate, normalised
    data.yaml                       Ultralytics dataset config (all sources)
    data-oi.yaml                    same, but its test split is Open Images only:
                                    the honest "photos in the wild" score
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data"
# Roboflow split name -> YOLO split name.
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = raw / f"{split}.zip"
        if target.exists():
            print(f"  {target.name} already downloaded")
            continue
        print(f"  downloading {split}.zip ...")
        urllib.request.urlretrieve(f"{HF_BASE}/{split}.zip", target)


def extract(raw: Path, extracted: Path) -> None:
    for split in SPLITS:
        dest = extracted / split
        if dest.exists() and any(dest.rglob("*.json")):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw / f"{split}.zip") as z:
            # Refuse entries that would escape the destination folder.
            for member in z.namelist():
                if not (dest / member).resolve().is_relative_to(dest.resolve()):
                    raise ValueError(f"unsafe path in {split}.zip: {member}")
            z.extractall(dest)


def convert_split(src: Path, out: Path, split: str) -> tuple[int, int]:
    """COCO json -> YOLO txt. Returns (images, boxes)."""
    ann_file = next(src.rglob("_annotations.coco.json"))
    coco = json.loads(ann_file.read_text(encoding="utf-8"))

    # Roboflow exports a parent "supercategory" (id 0) plus the real class; every
    # real category becomes class 0, since there is only one: license_plate.
    plate_ids = {c["id"] for c in coco["categories"] if c.get("supercategory") != "none"}

    images = {img["id"]: img for img in coco["images"]}
    boxes: dict[int, list[list[float]]] = {i: [] for i in images}
    for a in coco["annotations"]:
        if a["category_id"] in plate_ids and a["image_id"] in images:
            boxes[a["image_id"]].append(a["bbox"])

    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_boxes = 0
    for image_id, img in images.items():
        w, h = img["width"], img["height"]
        src_img = ann_file.parent / img["file_name"]
        shutil.copy2(src_img, img_dir / src_img.name)
        lines = []
        for x, y, bw, bh in boxes[image_id]:
            # Clamp to the image and drop degenerate boxes.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        (lbl_dir / f"{src_img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_boxes += len(lines)
    return len(images), n_boxes


OI_CLASS = "Vehicle registration plate"
# Open Images split -> YOLO split. Validation and test are small, so take them whole.
OI_SPLITS = {"train": "train", "validation": "val", "test": "test"}


def add_open_images(out: Path, max_train: int, splits: list[str]) -> dict[str, tuple[int, int]]:
    """Append Open Images V7 plate images to the YOLO folders. Returns per-split counts."""
    import fiftyone.zoo as foz  # optional dependency, imported only when asked for

    counts = {}
    for oi_split, yolo_split in OI_SPLITS.items():
        if oi_split not in splits:
            continue
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=oi_split,
            label_types=["detections"],
            classes=[OI_CLASS],
            only_matching=True,
            max_samples=max_train if oi_split == "train" else None,
            dataset_name=f"plates-oi-{oi_split}",
            overwrite=True,  # safe to re-run in the same Colab session
        )
        n_img = n_box = 0
        for sample in ds.iter_samples():
            dets = sample["ground_truth"].detections if sample["ground_truth"] else []
            boxes = [d.bounding_box for d in dets if d.label == OI_CLASS]
            if not boxes:
                continue
            src = Path(sample.filepath)
            name = f"oi_{src.stem}"
            shutil.copy2(src, out / "images" / yolo_split / f"{name}{src.suffix}")
            # FiftyOne boxes are [x, y, w, h], already normalised to [0, 1].
            lines = [f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}" for x, y, w, h in boxes if w > 0 and h > 0]
            (out / "labels" / yolo_split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_img += 1
            n_box += len(lines)
        counts[yolo_split] = (n_img, n_box)
    return counts


def write_yaml(path: Path, root: Path, test: str) -> None:
    path.write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: {test}\nnames:\n  0: license_plate\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"), help="root for raw/, extracted/ and yolo/")
    parser.add_argument("--skip-download", action="store_true", help="use zips already in <data>/raw")
    parser.add_argument("--open-images", type=int, default=0, metavar="N", help="also add up to N Open Images training images (0 = off)")
    parser.add_argument(
        "--open-images-splits",
        default="train,validation,test",
        help="Open Images splits to add; 'test' alone gives the in-the-wild score without the 2 GB training metadata",
    )
    args = parser.parse_args()

    raw, extracted, out = args.data / "raw", args.data / "extracted", args.data / "yolo"
    if not args.skip_download:
        print("Downloading dataset")
        download(raw)
    print("Extracting")
    extract(raw, extracted)

    if out.exists():
        shutil.rmtree(out)
    print("Converting COCO -> YOLO")
    for rf_split, yolo_split in SPLITS.items():
        n_img, n_box = convert_split(extracted / rf_split, out, yolo_split)
        print(f"  {yolo_split:5s}: {n_img:5d} images, {n_box:5d} plates")

    write_yaml(out / "data.yaml", out, "images/test")

    if args.open_images:
        print(f"Adding Open Images V7 '{OI_CLASS}' (up to {args.open_images} training images)")
        splits = [s.strip() for s in args.open_images_splits.split(",") if s.strip()]
        for split, (n_img, n_box) in add_open_images(out, args.open_images, splits).items():
            print(f"  {split:5s}: +{n_img:5d} images, +{n_box:5d} plates")
        # Ultralytics accepts a .txt list of image paths as a split.
        oi_test = sorted(p.resolve().as_posix() for p in (out / "images" / "test").glob("oi_*"))
        (out / "test-oi.txt").write_text("\n".join(oi_test) + "\n", encoding="utf-8")
        write_yaml(out / "data-oi.yaml", out, "test-oi.txt")
        print(f"Wrote {out / 'data-oi.yaml'} ({len(oi_test)} Open Images test images)")

    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
