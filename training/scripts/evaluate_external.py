"""Score the served models on an outside dataset they never trained on (e.g. Ghanaian plates).

Takes one or more Roboflow exports (zips or extracted folders, "YOLOv8"/"YOLOv11" format),
merges every split into a single test set and scores each ONNX file the app serves, at the
input size the app uses. Nothing here is used for training.

- Plate classes are picked by name (plate / licence / number / registration); other classes
  such as "car" are dropped. Polygon labels are converted to boxes.
- Duplicates are removed: Roboflow forks re-share the same photos, and its augmented training
  copies share the original's file stem (``IMG_1_jpg.rf.<hash>.jpg``). Unaugmented valid/test
  copies win over training ones.
- "clean" scores only images that came from valid/test splits, which Roboflow never augments.

    python training/scripts/evaluate_external.py --name ghana ~/Downloads/ghana-a.zip ~/Downloads/ghana-b.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO

PLATE = re.compile(r"plate|licen[cs]e|number|registration|\blp\b", re.I)
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ORDER = {"test": 0, "valid": 1, "val": 1, "train": 2}


def unpack(source: Path, raw: Path) -> Path:
    """Extracts a zip (once) into raw/<name>; folders are used in place."""
    if source.is_dir():
        return source
    dest = raw / source.stem
    if not dest.exists():
        with zipfile.ZipFile(source) as z:
            for member in z.namelist():
                # Only images, labels and the dataset yaml; refuse paths that escape dest.
                target = (dest / member).resolve()
                if not str(target).startswith(str(dest.resolve())) or member.endswith("/"):
                    continue
                if Path(member).suffix.lower() in IMAGE_TYPES | {".txt", ".yaml"}:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(member))
    return dest


def plate_classes(root: Path) -> set[int]:
    names = yaml.safe_load(next(root.rglob("data.yaml")).read_text(encoding="utf-8"))["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    keep = {i for i, n in names.items() if PLATE.search(str(n))}
    if not keep and len(names) == 1:
        keep = set(names)
    if not keep:
        raise SystemExit(f"{root}: no plate class among {names}")
    print(f"  {root.name}: classes {names} -> using {sorted(keep)}")
    return keep


def boxes(label: Path, keep: set[int]) -> list[str]:
    """YOLO box lines for plate classes; polygons become their bounding boxes."""
    out = []
    for line in label.read_text(encoding="utf-8").split("\n") if label.exists() else []:
        v = line.split()
        if len(v) < 5 or int(v[0]) not in keep:
            continue
        c = [float(x) for x in v[1:]]
        if len(c) > 4:
            xs, ys = c[0::2], c[1::2]
            c = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
        out.append("0 " + " ".join(f"{x:.6f}" for x in c))
    return out


def collect(roots: list[Path]) -> list[tuple[Path, list[str], bool]]:
    """(image, label lines, unaugmented) for every unique image across all datasets."""
    found = []
    for root in roots:
        keep = plate_classes(root)
        for image in root.rglob("*"):
            if image.suffix.lower() not in IMAGE_TYPES or image.parent.name != "images":
                continue
            split = image.parent.parent.name.lower()
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            found.append((SPLIT_ORDER.get(split, 2), image, boxes(label, keep), split != "train"))
    found.sort(key=lambda f: f[0])
    seen_stems, seen_hashes, unique = set(), set(), []
    for _, image, lines, clean in found:
        stem = image.name.split(".rf.")[0]
        digest = hashlib.md5(image.read_bytes()).hexdigest()
        if stem in seen_stems or digest in seen_hashes:
            continue
        seen_stems.add(stem)
        seen_hashes.add(digest)
        unique.append((image, lines, clean))
    return unique


def write_set(out: Path, items: list[tuple[Path, list[str], bool]]) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / "test").mkdir(parents=True)
    (out / "labels" / "test").mkdir(parents=True)
    for i, (image, lines, _) in enumerate(items):
        name = f"{i:05d}{image.suffix.lower()}"
        shutil.copyfile(image, out / "images" / "test" / name)
        (out / "labels" / "test" / f"{i:05d}.txt").write_text("\n".join(lines), encoding="utf-8")
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images/test", "val": "images/test", "test": "images/test", "names": {0: "plate"}}), encoding="utf-8")
    return yaml_path


def score(model: Path, data: Path, size: int) -> dict:
    m = YOLO(str(model), task="detect").val(data=str(data), split="test", imgsz=size, device="cpu", plots=False, verbose=False)
    return {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", type=Path, nargs="+", help="Roboflow export zips or folders")
    p.add_argument("--name", default="ghana", help="name for the test set and results file")
    p.add_argument("--model-dir", type=Path, default=Path("web/public/model"))
    p.add_argument("--data", type=Path, default=Path("data/external"))
    p.add_argument("--results", type=Path, default=Path("training/results"))
    args = p.parse_args()

    raw = args.data / args.name / "raw"
    roots = [unpack(s, raw) for s in args.sources]
    items = collect(roots)
    clean = [it for it in items if it[2]]
    plates = sum(len(lines) for _, lines, _ in items)
    print(f"  {len(items)} unique images ({len(clean)} unaugmented), {plates} plates")

    sets = {"all": write_set(args.data / args.name / "all", items)}
    if clean and len(clean) < len(items):
        sets["clean"] = write_set(args.data / args.name / "clean", clean)

    manifest = json.loads((args.model_dir / "models.json").read_text(encoding="utf-8"))
    results = {"images": len(items), "unaugmented": len(clean), "plates": plates, "sources": [s.name for s in args.sources], "variants": {}}
    for variant, meta in manifest["variants"].items():
        model = args.model_dir / meta["file"]
        print(f"[{variant}] {model.name} at {meta['imgsz']}px")
        results["variants"][variant] = {"file": model.name, "imgsz": meta["imgsz"], **{k: score(model, d, meta["imgsz"]) for k, d in sets.items()}}
        print("  ", json.dumps(results["variants"][variant]))

    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"{args.name}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
