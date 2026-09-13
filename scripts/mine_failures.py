# Finds the test pages the model got most wrong and renders them side by
# side with ground truth. The 5 failure cases in the memo come from this,
# not from guessing - the brief flags "no acknowledged failure cases" as a
# red flag, so these need to be real, pointable-at examples.
#
# Scoring is deliberately crude (greedy IoU 0.5 matching) - just needs to
# rank pages well enough to surface the interesting ones, not be a precise
# metric (that's evaluate.py's job).
#
# Usage:
#   python scripts/mine_failures.py --weights runs/detect/.../best.pt \
#       --data data/doclaynet --out reports/failures --top 25
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.constants import ID_TO_CLASS  # noqa: E402
from scripts._render_utils import load_label_font as _label_font  # noqa: E402

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
]


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _read_ground_truth(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, ...]]]:
    """Reads a YOLO label file back into absolute xyxy boxes."""
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, xc, yc, w, h = line.split()
        xc, yc, w, h = float(xc) * width, float(yc) * height, float(w) * width, float(h) * height
        boxes.append((int(class_id), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
    return boxes


def score_page(
    ground_truth: list[tuple[int, tuple[float, ...]]],
    predictions: list[tuple[int, tuple[float, ...], float]],
    iou_threshold: float = 0.5,
) -> dict:
    """Greedy match + counts. Splits misses from misclassifications since
    they have different causes - a miss is usually small/thin/query-
    starved, a misclassification is usually genuine label ambiguity
    (Text vs List-item, Title vs Section-header). That split is what lets
    the memo talk about annotation ambiguity instead of just "missed stuff"."""
    unmatched_gt = list(range(len(ground_truth)))
    matched_predictions: set[int] = set()

    correct = 0
    misclassified = 0

    for gt_index in list(unmatched_gt):
        gt_class, gt_box = ground_truth[gt_index]

        best_index, best_iou = None, iou_threshold
        for pred_index, (_, pred_box, _) in enumerate(predictions):
            if pred_index in matched_predictions:
                continue
            overlap = iou(gt_box, pred_box)
            if overlap >= best_iou:
                best_index, best_iou = pred_index, overlap

        if best_index is not None:
            matched_predictions.add(best_index)
            unmatched_gt.remove(gt_index)
            if predictions[best_index][0] == gt_class:
                correct += 1
            else:
                misclassified += 1

    false_positives = len(predictions) - len(matched_predictions)

    return {
        "ground_truth_regions": len(ground_truth),
        "predicted_regions": len(predictions),
        "correct": correct,
        "misclassified": misclassified,
        "missed": len(unmatched_gt),
        "false_positives": false_positives,
        # misclassifications weighted higher - they're the interesting ones
        "error_score": len(unmatched_gt) + false_positives + 1.5 * misclassified,
    }


def _draw_labelled_box(draw, colour: str, x1, y1, x2, y2, label: str, font) -> None:
    """Box outline plus a solid background behind the label, legible over
    any page content."""
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=4)
    text_box = draw.textbbox((x1, y1), label, font=font)
    draw.rectangle(
        [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
        fill=colour,
    )
    draw.text((x1, y1), label, font=font, fill="white")


def _render_comparison(image, ground_truth, predictions, out_path: Path) -> None:
    """Ground truth on the left, prediction on the right."""
    from PIL import Image, ImageDraw

    width, height = image.size
    canvas = Image.new("RGB", (width * 2 + 20, height + 40), "white")
    canvas.paste(image, (0, 40))
    canvas.paste(image, (width + 20, 40))

    draw = ImageDraw.Draw(canvas)
    header_font = _label_font(24)
    label_font = _label_font(22)
    draw.text((10, 8), "GROUND TRUTH", font=header_font, fill="black")
    draw.text((width + 30, 8), "PREDICTION", font=header_font, fill="black")

    for class_id, (x1, y1, x2, y2) in ground_truth:
        colour = PALETTE[class_id % len(PALETTE)]
        _draw_labelled_box(draw, colour, x1, y1 + 40, x2, y2 + 40,
                            ID_TO_CLASS.get(class_id, "?"), label_font)

    offset = width + 20
    for class_id, (x1, y1, x2, y2), confidence in predictions:
        colour = PALETTE[class_id % len(PALETTE)]
        label = f"{ID_TO_CLASS.get(class_id, '?')} {confidence:.2f}"
        _draw_labelled_box(draw, colour, x1 + offset, y1 + 40, x2 + offset, y2 + 40,
                            label, label_font)

    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="dataset root containing images/ and labels/")
    parser.add_argument("--out", default="reports/failures")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    from PIL import Image
    from ultralytics import RTDETR

    data_root = Path(args.data)
    image_dir = data_root / "images" / "test"
    label_dir = data_root / "labels" / "test"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = RTDETR(args.weights)
    images = sorted(image_dir.glob("*.png"))
    print(f"Scoring {len(images)} test pages...")

    manifest_path = data_root / "manifest_test.json"
    categories = {}
    if manifest_path.exists():
        categories = {
            row["stem"]: row["doc_category"]
            for row in json.loads(manifest_path.read_text(encoding="utf-8"))
        }

    scored = []
    for index, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        result = model.predict(image, conf=args.conf, verbose=False)[0]
        predictions = [
            (int(cls), tuple(float(v) for v in box), float(conf))
            for cls, box, conf in zip(
                result.boxes.cls.tolist(),
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
            )
        ]

        ground_truth = _read_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        score = score_page(ground_truth, predictions)
        score["stem"] = image_path.stem
        score["doc_category"] = categories.get(image_path.stem, "unknown")
        scored.append((score, image, ground_truth, predictions))

        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(images)}", flush=True)

    scored.sort(key=lambda item: item[0]["error_score"], reverse=True)

    summary = []
    for rank, (score, image, ground_truth, predictions) in enumerate(scored[: args.top], start=1):
        out_path = out_dir / f"{rank:02d}_{score['stem']}_err{score['error_score']:.0f}.png"
        _render_comparison(image, ground_truth, predictions, out_path)
        score["render"] = out_path.name
        summary.append(score)

    (out_dir / "failure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nWrote {len(summary)} comparison renders to {out_dir}")
    print("\nWorst pages:")
    for score in summary[:8]:
        print(
            f"  {score['stem']:20s} {score['doc_category']:20s} "
            f"missed={score['missed']:3d} misclassified={score['misclassified']:3d} "
            f"fp={score['false_positives']:3d}"
        )


if __name__ == "__main__":
    main()
