# Converts the HF DocLayNet-base dataset into the image/label layout
# Ultralytics expects. Everything that can go wrong here goes wrong
# silently - bad labels don't raise, training just quietly learns the
# wrong thing. Three things beyond the obvious conversion:
#   1. collapses repeated per-line boxes (a 40-region page can show up as
#      ~1,100 duplicate annotations otherwise)
#   2. drops degenerate boxes and counts them (zero-area -> NaN loss later)
#   3. can render a sample with decoded labels drawn on, so I can eyeball
#      that the class index mapping is actually right
#
# Usage:
#   python scripts/prepare_dataset.py --out data/doclaynet
#   python scripts/prepare_dataset.py --out data/doclaynet --verify 12
#   python scripts/prepare_dataset.py --out data/doclaynet --limit 20   # smoke test
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import CLASS_NAMES, ID_TO_CLASS  # noqa: E402

HF_DATASET = "pierreguillou/DocLayNet-base"

# HF split names don't match Ultralytics' directory convention
SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def coco_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """DocLayNet gives [x, y, w, h] top-left in pixels. Ultralytics wants
    [x_centre, y_centre, w, h] normalised 0-1. Easy to get the corner-to-
    centre shift wrong without noticing - tested for that reason."""
    x, y, w, h = bbox
    x_centre = (x + w / 2) / img_w
    y_centre = (y + h / 2) / img_h
    return (x_centre, y_centre, w / img_w, h / img_h)


def is_valid_bbox(bbox: list[float], img_w: int, img_h: int) -> bool:
    """Rejects zero-area/out-of-bounds boxes before they turn into NaN
    losses several epochs in. Counts rejections rather than just dropping
    silently."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    if x + w > img_w or y + h > img_h:
        return False
    return True


def dedupe_annotations(
    bboxes: list[list[float]], categories: list[int]
) -> list[tuple[tuple[float, ...], int]]:
    """DocLayNet-base repeats a block's box once per text line inside it -
    a paragraph with 6 lines shows up as 6 identical entries. Dedupe on
    (box, category) rather than box alone, since two classes occasionally
    share the same extent. First-seen order kept so output is reproducible."""
    if len(bboxes) != len(categories):
        raise ValueError(
            f"bboxes and categories must be the same length, "
            f"got {len(bboxes)} and {len(categories)}. "
            "If this fires, the dataset schema is not what this script assumes."
        )

    seen: set[tuple[tuple[float, ...], int]] = set()
    unique: list[tuple[tuple[float, ...], int]] = []

    for bbox, category in zip(bboxes, categories):
        key = (tuple(float(v) for v in bbox), int(category))
        if key not in seen:
            seen.add(key)
            unique.append(key)

    return unique


def _write_split(
    dataset, split_name: str, out_dir: Path, limit: int | None
) -> dict:
    """Writes one split to disk, returns the counts (these end up in the
    memo, especially the dedupe ratio - "removed 96% of raw annotations"
    needs a number next to it or it sounds like something broke)."""
    image_dir = out_dir / "images" / split_name
    label_dir = out_dir / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rows = dataset[split_name]
    total = len(rows) if limit is None else min(limit, len(rows))

    stats = {
        "images": 0,
        "raw_annotations": 0,
        "after_dedupe": 0,
        "dropped_invalid": 0,
        "empty_pages": 0,
        "class_counts": Counter(),
        "doc_category_counts": Counter(),
    }

    # per-image manifest for evaluate.py's per-category mAP breakdown, and
    # for measuring train/test source-PDF leakage
    manifest: list[dict] = []

    for index in range(total):
        row = rows[index]
        image = row["image"]
        img_w, img_h = image.size

        stats["raw_annotations"] += len(row["bboxes_block"])

        annotations = dedupe_annotations(row["bboxes_block"], row["categories"])
        stats["after_dedupe"] += len(annotations)

        lines = []
        for bbox, category in annotations:
            if not is_valid_bbox(list(bbox), img_w, img_h):
                stats["dropped_invalid"] += 1
                continue
            xc, yc, w, h = coco_to_yolo(list(bbox), img_w, img_h)
            lines.append(f"{category} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            stats["class_counts"][ID_TO_CLASS.get(category, f"UNKNOWN_{category}")] += 1

        # blank page is legit, but also what a conversion bug looks like
        if not lines:
            stats["empty_pages"] += 1

        stem = f"{split_name}_{index:06d}"
        image.convert("RGB").save(image_dir / f"{stem}.png")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        doc_category = row.get("doc_category", "unknown")
        stats["doc_category_counts"][doc_category] += 1
        stats["images"] += 1

        manifest.append({
            "stem": stem,
            "doc_category": doc_category,
            "source_pdf": row.get("original_filename", "unknown"),
            "num_regions": len(lines),
        })

        if stats["images"] % 500 == 0:
            print(f"  {split_name}: {stats['images']}/{total}", flush=True)

    (out_dir / f"manifest_{split_name}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    stats["class_counts"] = dict(stats["class_counts"])
    stats["doc_category_counts"] = dict(stats["doc_category_counts"])
    return stats


def _render_label_check(dataset, out_dir: Path, sample_size: int) -> None:
    """Draws decoded class labels on a sample of pages so I can eyeball
    them. My class ordering (0-indexed alphabetical) is inferred, not
    confirmed - if it's off by one, Table becomes Section-header
    everywhere and training won't complain about it. No automated test
    catches that, only looking does."""
    from PIL import ImageDraw

    from scripts._render_utils import load_label_font

    label_font = load_label_font(28)

    check_dir = out_dir / "label_check"
    check_dir.mkdir(parents=True, exist_ok=True)

    rows = dataset["train"]
    random.seed(42)
    indices = random.sample(range(len(rows)), min(sample_size, len(rows)))

    palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
        "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    ]

    for index in indices:
        row = rows[index]
        image = row["image"].convert("RGB")
        draw = ImageDraw.Draw(image)

        for bbox, category in dedupe_annotations(row["bboxes_block"], row["categories"]):
            x, y, w, h = bbox
            colour = palette[category % len(palette)]
            label = ID_TO_CLASS.get(category, "?")
            draw.rectangle([x, y, x + w, y + h], outline=colour, width=4)

            # solid background behind the label - plain text is invisible
            # on dark scanned content otherwise
            text_box = draw.textbbox((x, y), label, font=label_font)
            draw.rectangle(
                [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
                fill=colour,
            )
            draw.text((x, y), label, font=label_font, fill="white")

        image.save(check_dir / f"check_{index:06d}.png")

    print(f"\nWrote {len(indices)} annotated pages to {check_dir}")
    print("LOOK AT THESE before training. If the box labelled 'Table' is not")
    print("drawn around a table, the class index base is wrong and everything")
    print("downstream will be quietly meaningless.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/doclaynet", help="output root")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap images per split - for smoke tests only, not for a real run",
    )
    parser.add_argument(
        "--verify", type=int, default=0,
        help="render this many annotated pages for visual label checking",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import ClassLabel, Features, Sequence, Value
    from datasets import load_dataset
    from datasets.features import Image as HFImage

    # The dataset's own loading script declares these as int64, but the
    # real coordinates are floats (hit one directly: 139.664355). Newer
    # pyarrow refuses that lossy cast where older versions silently
    # floored it. Can't edit someone else's script, so override the
    # schema on load instead - copied field-for-field, only these two
    # fixed to float64.
    doclaynet_features = Features({
        "id": Value("string"),
        "texts": Sequence(Value("string")),
        "bboxes_block": Sequence(Sequence(Value("float64"))),
        "bboxes_line": Sequence(Sequence(Value("float64"))),
        "categories": Sequence(ClassLabel(names=CLASS_NAMES)),
        "image": HFImage(),
        "page_hash": Value("string"),
        "original_filename": Value("string"),
        "page_no": Value("int32"),
        "num_pages": Value("int32"),
        "original_width": Value("int32"),
        "original_height": Value("int32"),
        "coco_width": Value("int32"),
        "coco_height": Value("int32"),
        "collection": Value("string"),
        "doc_category": Value("string"),
    })

    print(f"Loading {HF_DATASET} (3.8 GB on first run, cached after)...", flush=True)
    try:
        # trust_remote_code=True: this repo ships a loading script rather
        # than static parquet. Without this it prompts for confirmation,
        # which hangs forever under Kaggle's non-interactive commit runs.
        # Checked what the script does first - it's just the author's own
        # DocLayNet -> HF datasets conversion, nothing else.
        dataset = load_dataset(HF_DATASET, trust_remote_code=True, features=doclaynet_features)
    except RuntimeError as error:
        # datasets>=4.0.0 dropped loading-script support entirely -
        # requirements.txt pins <4.0.0 for this reason, but if an
        # environment already has a newer version cached, this is what
        # they'll hit.
        if "no longer supported" in str(error):
            raise RuntimeError(
                f"{error}\n\n"
                "This dataset repo uses the old Hugging Face 'loading script' "
                "format, which datasets>=4.0.0 removed support for entirely. "
                "Fix: pip install \"datasets<4.0.0\" (already pinned in "
                "requirements.txt - your environment likely has a newer "
                "version cached from something else)."
            ) from error
        raise

    if args.verify:
        _render_label_check(dataset, out_dir, args.verify)

    report = {"dataset": HF_DATASET, "splits": {}}
    for hf_split, dir_name in SPLITS.items():
        print(f"Converting split '{hf_split}'...", flush=True)
        report["splits"][dir_name] = _write_split(dataset, hf_split, out_dir, args.limit)

    _write_data_yaml(out_dir)

    report_path = out_dir / "prep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote {report_path}")
    for split, stats in report["splits"].items():
        raw, kept = stats["raw_annotations"], stats["after_dedupe"]
        shrink = (1 - kept / raw) * 100 if raw else 0
        print(
            f"  {split:11s} {stats['images']:5d} images | "
            f"{raw:7d} raw -> {kept:6d} annotations ({shrink:.1f}% were repeats) | "
            f"{stats['dropped_invalid']} invalid dropped"
        )


def _write_data_yaml(out_dir: Path) -> None:
    """Generates the Ultralytics config instead of hand-writing it, so
    class names can't drift from app/constants.py."""
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml = (
        "# Generated by scripts/prepare_dataset.py - do not edit by hand.\n"
        "# Class names come from app/constants.py so the two cannot drift.\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/validation\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"{names}\n"
    )
    (out_dir / "doclaynet.yaml").write_text(yaml, encoding="utf-8")


if __name__ == "__main__":
    main()
