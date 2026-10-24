"""Adds Ghanaian sample photos to the app (web/public/samples), with their labelled plates.

The photos come from the public-domain Ghanaian License Plates dataset, which the models never
trained on; plate numbers come from training/ocr/ghana-plates.csv. Re-running replaces them.

    python training/scripts/add_ghana_samples.py "path/to/Ghanaian License Plates.v2i.yolov11.zip"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_external import unpack  # noqa: E402

# One clear plate each, in different styles: one line, two lines, yellow commercial, old letter suffix.
PLATES = ["GW 4166-23", "GG 3096-17", "GB 3487-21", "GT 9556 V"]
FIRST_ID = 101  # keeps them apart from the original samples (1-8)
SIZES = {"src": 1024, "hero": 840, "thumb": 320}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--labels", type=Path, default=Path("training/ocr/ghana-plates.csv"))
    p.add_argument("--out", type=Path, default=Path("web/public/samples"))
    args = p.parse_args()

    files = {f.name: f for f in unpack(args.dataset, Path("data/external/ghana/raw")).rglob("*.jpg")}
    rows: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(args.labels, encoding="utf-8")):
        rows.setdefault(r["image"], []).append(r)

    manifest_path = args.out / "samples.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"] = [s for s in manifest["samples"] if s["id"] < FIRST_ID]
    added = []
    for n, plate in enumerate(PLATES):
        # The first photo with exactly this one plate in it.
        name = next(k for k, rs in rows.items() if len(rs) == 1 and rs[0]["plate"] == plate)
        r = rows[name][0]
        im = Image.open(files[name]).convert("RGB")
        sid = FIRST_ID + n
        paths = {}
        for kind, size in SIZES.items():
            out = im.copy()
            out.thumbnail((size, size), Image.LANCZOS)
            ext = "jpg" if kind == "src" else "webp"
            rel = f"samples/sample-{sid}{'' if kind == 'src' else '-' + kind}.{ext}"
            out.save(args.out.parent / rel, quality=85 if ext == "jpg" else 80)
            paths[kind] = rel
        W, H = im.size
        cx, cy, w, h = (float(r[k]) for k in ("cx", "cy", "w", "h"))
        added.append({
            "id": sid,
            "src": paths["src"],
            "thumb": paths["thumb"],
            "width": W,
            "height": H,
            "truth": [{"x": round((cx - w / 2) * W, 1), "y": round((cy - h / 2) * H, 1), "w": round(w * W, 1), "h": round(h * H, 1), "plate": plate}],
            "hero": paths["hero"],
            "source": "Ghanaian License Plates (luther-eyandi, Roboflow Universe), public domain",
        })
        print(sid, plate, name)
    manifest["samples"] = added + manifest["samples"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
