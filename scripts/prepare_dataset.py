"""
Turns the Hugging Face DocLayNet-base rows into the image/label layout that
Ultralytics expects on disk.

This script ended up being the most careful piece of code in the project, and
not because the conversion maths is hard. It is because everything that can go
wrong here goes wrong *silently*. Bad labels do not raise. Training runs to
completion, the loss curve looks normal, and the only symptom is that the final
numbers are worse than they should be for reasons you cannot see. By the time
you notice, you have burned your GPU quota.

So there are three things this file does beyond the obvious conversion, each of
which came from a specific worry:

1. It collapses the repeated bounding boxes. DocLayNet-base stores the parent
   block's box once per text line, so a six-line paragraph shows up as six
   identical annotations. Writing those out raw inflates a 40-region page into
   a ~1,100-annotation page and teaches the model a nonsense prior about object
   density.

2. It drops degenerate boxes and counts them. Zero-area labels turn into NaN
   losses several epochs into training, which is a horrible thing to debug.

3. It can render a sample of pages with the decoded class names drawn on, so I
   can actually look at them and confirm the category indices mean what I think
   they mean. I did not want to take the class ordering on trust from a dataset
   card - if that is off by one, every number I report afterwards is fiction.

Everything it did is written to prep_report.json so the counts end up in the
memo instead of being lost.

Usage:
    python scripts/prepare_dataset.py --out data/doclaynet
    python scripts/prepare_dataset.py --out data/doclaynet --verify 12
    python scripts/prepare_dataset.py --out data/doclaynet --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# The class list lives in app/constants.py and nowhere else. See the note in
# that file about why I stopped duplicating it.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import CLASS_NAMES, ID_TO_CLASS  # noqa: E402

HF_DATASET = "pierreguillou/DocLayNet-base"

# The Hugging Face split names do not match the directory names Ultralytics
# conventionally uses, so I map them explicitly rather than relying on both
# sides happening to agree.
SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def coco_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """
    Converts one box from DocLayNet's format into the one Ultralytics wants.

    DocLayNet gives [x, y, w, h] in absolute pixels, measured from the top-left
    corner of the page. Ultralytics wants [x_centre, y_centre, w, h] normalised
    into 0-1.

    The part worth being careful about is the corner-to-centre shift. If you
    forget it, every box lands half its own size up and to the left. That is
    subtle enough that it does not look like a bug - it looks like the model is
    slightly imprecise at localisation - which is exactly why I tested it.
    """
    x, y, w, h = bbox
    x_centre = (x + w / 2) / img_w
    y_centre = (y + h / 2) / img_h
    return (x_centre, y_centre, w / img_w, h / img_h)


def is_valid_bbox(bbox: list[float], img_w: int, img_h: int) -> bool:
    """
    Rejects boxes that would poison training rather than letting them through.

    Zero-area boxes are the dangerous ones: Ultralytics accepts them and then
    produces NaN losses some epochs later, well after you have stopped watching.
    Out-of-bounds boxes are less lethal but they normalise to values outside
    0-1, which quietly breaks the loss computation.

    I count every rejection rather than just dropping it, because "how clean was
    your data" is a question I would rather answer with a number.
    """
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
    """
    Collapses the per-line repetition in DocLayNet-base down to one entry per
    actual region.

    This is the single most important function in the file. The dataset stores
    `bboxes_block` aligned to text lines, so the block box for a paragraph is
    repeated once for every line inside it. A dense page can carry over a
    thousand entries describing perhaps forty real regions.

    I dedupe on the (box, category) pair rather than the box alone, because very
    occasionally two different classes are labelled over the same extent and
    throwing one away would be losing real signal.

    I keep first-seen order so the label files come out byte-identical between
    runs, which makes it easy to diff the effect of a change to this script.
    """
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
    """
    Writes one split to disk and reports exactly what it did.

    The counts returned here are the ones that end up in the memo. I wanted the
    deduplication figure in particular to be visible, because "I removed 96% of
    the raw annotations" is a claim that needs a number attached to it or it
    sounds like I broke something.
    """
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

    """
    I keep a per-image manifest as well as the aggregate counts.

    The counts alone tell me the dataset is balanced; the manifest is what lets
    evaluate.py split mAP by document category afterwards. I need that because
    one aggregate number cannot tell me whether the model learned document
    structure or just learned what financial reports look like - and financial
    reports are the biggest slice of DocLayNet, so a model that is good at those
    and useless on patents would still post a respectable headline score.

    I also record the source PDF filename here. DocLayNet pages come from
    multi-page documents, so if pages from the same PDF ended up on both sides
    of the train/test boundary my test numbers are optimistic. I would rather
    measure that overlap and report it than assume it is zero.
    """
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

        # A page with no surviving annotations is legitimate (a blank page) but
        # it is also what a conversion bug looks like, so I count them.
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
    """
    Draws boxes with their decoded class names onto a handful of pages so I can
    look at them with my own eyes.

    This exists because of one specific fear. The class ordering I am using is
    0-indexed alphabetical, which I inferred rather than found stated
    unambiguously. If it is off by even one, `Table` becomes `Section-header`
    everywhere, training proceeds perfectly happily, and every metric and every
    failure-case analysis I write afterwards is describing a model that learned
    something other than what I claim.

    No automated test can catch that - the data is self-consistent either way.
    The only check that works is rendering a page and seeing whether the box
    labelled "Table" is actually drawn around a table. So I look before I train.
    """
    from PIL import ImageDraw

    from scripts._render_utils import load_label_font

    label_font = load_label_font(28)

    # Writing straight into out_dir rather than climbing to its parent, same
    # as prep_report.json and doclaynet.yaml below. My first version reached
    # up to out_dir.parent, which put this somewhere I then pointed the
    # notebook's check-cell at incorrectly - "up one, then back down two"
    # is exactly the kind of path arithmetic that is easy to get wrong once
    # and never notice, because the directory still gets created either way.
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

            """
            Plain coloured text sitting directly on a document page is
            unreadable half the time - the page is mostly white, but a label
            landing on dark scanned text or another box's fill is invisible.
            I draw a solid rectangle behind the label first, sized to the
            actual text, so it reads the same regardless of what is under it.
            """
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

    """
    The dataset repo's own loading script declares bboxes_block/bboxes_line
    as Sequence(Sequence(Value("int64"))). That is simply wrong - the actual
    coordinates in the underlying data are floats (I hit one directly:
    139.664355). Older pyarrow used to silently floor a float into that int64
    slot; a newer pyarrow (which is what Kaggle ships) refuses the lossy cast
    outright and the whole load fails with `ArrowInvalid: Float value ...
    was truncated converting to int64`.

    I cannot edit someone else's script on the Hub, but `load_dataset` lets me
    override the schema it builds against. So I reconstruct the script's exact
    feature dict from source and correct only the two fields that are wrong,
    to float64. Everything else is left byte-for-byte identical to the
    original so I am not silently changing anything I have not verified needs
    changing.
    """
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
        """
        trust_remote_code=True is required because this dataset repo is the
        legacy "loading script" format: it ships a small Python file
        (DocLayNet-base.py) that datasets executes to build the dataset,
        rather than reading a static Parquet/Arrow file directly.

        Without this flag, `datasets` stops and asks for interactive
        confirmation before running someone else's code - a sensible default
        for a random dataset off the Hub. It also means the prompt blocks
        forever in a non-interactive run, which is exactly how I run this: the
        Kaggle notebook uses Save & Run All (Commit) so the session survives
        me closing the browser, and there is no terminal on the other end to
        type "y" into.

        I am passing it explicitly rather than just suppressing the prompt,
        because I looked at what the script does before trusting it: it is
        the dataset author's own conversion of IBM's DocLayNet into the
        HF `datasets` structure, nothing more.
        """
        dataset = load_dataset(HF_DATASET, trust_remote_code=True, features=doclaynet_features)
    except RuntimeError as error:
        """
        This is the one dependency failure I actually hit while building this,
        so I am catching it by name rather than leaving the next person (which
        might be me, or a reviewer reproducing the run) to decode a stack trace.

        pierreguillou/DocLayNet-base ships as a legacy "loading script" dataset
        repo, and Hugging Face's `datasets` library removed loading-script
        support outright in 4.0.0. requirements.txt pins `datasets<4.0.0` for
        exactly this reason, but if someone's environment already has a newer
        version cached, this is what they hit.
        """
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
    """
    Emits the Ultralytics dataset config next to the data it describes.

    I generate this rather than hand-writing it so the class names in the YAML
    physically cannot drift from app/constants.py. That drift is the exact bug
    I was worried about when I consolidated the class list in the first place.
    """
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
