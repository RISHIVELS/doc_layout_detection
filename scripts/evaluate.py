# Evaluates the trained model on the held-out test split. One aggregate
# mAP can't answer the questions that actually matter, so this produces:
#   - per-class AP (which classes are actually weak, and was that
#     predictable given the 640px downscale)
#   - per-document-category mAP (did it learn structure, or just what
#     financial reports look like - the biggest slice of the data)
#   - query-budget saturation rate (RT-DETR has a fixed number of
#     predictions per image - dense pages lose recall for a reason that
#     has nothing to do with training quality)
#   - measured train/test source-PDF overlap (used the author's splits for
#     reproducibility, but that means inheriting whatever leakage they have)
#
# Usage:
#   python scripts/evaluate.py --weights runs/detect/rtdetr_doclaynet/weights/best.pt \
#                              --data data/doclaynet/doclaynet.yaml --out reports
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import CLASS_NAMES, DEFAULT_QUERY_BUDGET  # noqa: E402


def _load_manifest(data_root: Path, split: str) -> list[dict]:
    path = data_root / f"manifest_{split}.json"
    if not path.exists():
        print(f"  (no manifest at {path} - skipping category breakdown)")
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def measure_split_leakage(data_root: Path) -> dict:
    """Checks if any source PDF has pages on both sides of train/test.
    Used the author's published splits rather than re-splitting (keeps
    numbers comparable to published work), but that means inheriting
    whatever overlap they have. Not fixing it, just measuring and
    reporting it."""
    train = {row["source_pdf"] for row in _load_manifest(data_root, "train")}
    test_rows = _load_manifest(data_root, "test")
    if not train or not test_rows:
        return {"measured": False}

    shared = {row["source_pdf"] for row in test_rows} & train
    affected = sum(1 for row in test_rows if row["source_pdf"] in shared)

    return {
        "measured": True,
        "shared_source_pdfs": len(shared),
        "test_pages_from_shared_pdfs": affected,
        "test_pages_total": len(test_rows),
        "percent_affected": round(100 * affected / len(test_rows), 1),
    }


def measure_query_saturation(data_root: Path, budget: int = DEFAULT_QUERY_BUDGET) -> dict:
    """RT-DETR emits a fixed number of boxes per image regardless of
    confidence - a page with more regions than that structurally can't be
    fully detected. Measuring this separately so it doesn't get
    misattributed as a training/recall problem in the failure analysis."""
    rows = _load_manifest(data_root, "test")
    if not rows:
        return {"measured": False}

    counts = [row["num_regions"] for row in rows]
    saturated = [c for c in counts if c > budget]

    return {
        "measured": True,
        "query_budget": budget,
        "max_regions_on_any_page": max(counts),
        "mean_regions_per_page": round(sum(counts) / len(counts), 1),
        "pages_over_budget": len(saturated),
        "percent_over_budget": round(100 * len(saturated) / len(counts), 2),
    }


def per_category_map(weights: str, data_root: Path, base_yaml: Path, out_dir: Path) -> dict:
    """Runs validation once per document category by writing a filtered
    image-list yaml per category. Six extra small val passes, turns one
    opaque number into a real generalisation check."""
    rows = _load_manifest(data_root, "test")
    if not rows:
        return {}

    from ultralytics import RTDETR

    by_category: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        image_path = data_root / "images" / "test" / f"{row['stem']}.png"
        by_category[row["doc_category"]].append(str(image_path.resolve()))

    split_dir = out_dir / "category_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    base = base_yaml.read_text(encoding="utf-8")
    results: dict[str, dict] = {}

    for category, images in sorted(by_category.items()):
        # too few pages = noise, not a measurement - skip and say so
        if len(images) < 10:
            results[category] = {"pages": len(images), "skipped": "too few pages to be meaningful"}
            continue

        listing = split_dir / f"{category}.txt"
        listing.write_text("\n".join(images), encoding="utf-8")

        yaml_path = split_dir / f"{category}.yaml"
        yaml_path.write_text(
            base.replace("val: images/validation", f"val: {listing.resolve().as_posix()}"),
            encoding="utf-8",
        )

        metrics = RTDETR(weights).val(data=str(yaml_path), split="val", verbose=False)
        results[category] = {
            "pages": len(images),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
        }
        print(f"  {category:22s} {len(images):4d} pages  mAP50={metrics.box.map50:.3f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="path to doclaynet.yaml")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    from ultralytics import RTDETR

    base_yaml = Path(args.data)
    data_root = base_yaml.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating on the held-out test split...")
    metrics = RTDETR(args.weights).val(data=args.data, split="test", plots=True)

    # zip against my own class list rather than trusting box.maps' order -
    # a mismatch here would attribute every class's score to its neighbour
    # and still look completely plausible
    per_class = {
        name: round(float(ap), 4)
        for name, ap in zip(CLASS_NAMES, list(metrics.box.maps))
    }

    report = {
        "weights": args.weights,
        "overall": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class_mAP50_95": per_class,
        "query_saturation": measure_query_saturation(data_root),
        "split_leakage": measure_split_leakage(data_root),
        "confusion_matrix_plot": str(Path(metrics.save_dir) / "confusion_matrix_normalized.png"),
    }

    print("\nPer-document-category breakdown:")
    report["per_doc_category"] = per_category_map(args.weights, data_root, base_yaml, out_dir)

    path = out_dir / "metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote {path}")
    print(f"  overall mAP50     {report['overall']['mAP50']:.3f}")
    print(f"  overall mAP50-95  {report['overall']['mAP50_95']:.3f}")
    print("\n  weakest classes:")
    for name, ap in sorted(per_class.items(), key=lambda kv: kv[1])[:4]:
        print(f"    {name:16s} {ap:.3f}")


if __name__ == "__main__":
    main()
