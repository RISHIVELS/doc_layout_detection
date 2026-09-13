# Fine-tunes RT-DETR on the prepared DocLayNet data. Every hyperparameter
# is set explicitly (nothing left to Ultralytics defaults) and the run
# writes a metadata file with GPU, wall-clock time and library versions,
# since the brief caps Part A at 50% if the run isn't reproducible.
#
# Usage:
#   python scripts/train.py --data data/doclaynet/doclaynet.yaml
#   python scripts/train.py --data ... --epochs 1 --smoke   # CPU smoke test
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

RANDOM_SEED = 42


def _build_train_args(data: str, epochs: int, batch: int, imgsz: int, device: str) -> dict:
    """All the hyperparameters. The augmentation block is the part worth
    reading - three of Ultralytics' defaults are actively wrong for
    documents:
      - fliplr 0.5 mirrors pages, which don't have a valid mirrored layout
        (text direction, page numbers in fixed corners)
      - flipud same problem, more obviously (upside-down page)
      - mosaic composites 4 images into one, destroying whole-page
        structure (header at top, footer at bottom) that the model needs
    Kept small rotation + mild perspective - those model real scan/camera
    skew and are the only hedge against the hidden eval set containing
    photographed rather than rendered pages.

    lr0=1e-4 not the usual YOLO 1e-2 - DETR-family transformer components
    are unstable at high LR, and I'm fine-tuning from COCO weights, not
    training from scratch."""
    return {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "seed": RANDOM_SEED,
        "deterministic": True,

        "amp": True,  # needed to fit RT-DETR-L at batch 8 in 16GB on a T4

        "optimizer": "AdamW",
        "lr0": 1e-4,
        "lrf": 0.01,
        "weight_decay": 1e-4,
        "warmup_epochs": 3.0,
        "cos_lr": True,

        "save_period": 1,  # checkpoint every epoch - Kaggle sessions can drop
        "patience": 10,

        "workers": 2,          # Kaggle's few CPU cores - more workers stall
        "cache": False,        # 8k pages at 1025px won't fit in RAM
        "val": True,
        "plots": True,

        # --- augmentation, see docstring above ---
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.0,
        "degrees": 3.0,
        "perspective": 0.0005,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "hsv_h": 0.0,          # document colour isn't a useful signal
        "hsv_s": 0.2,
        "hsv_v": 0.2,          # scan brightness genuinely varies
        "erasing": 0.0,        # occluding part of a region changes its class
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _write_run_metadata(out_dir: Path, train_args: dict, seconds: float) -> None:
    """Everything needed to reproduce this run. GPU name matters most -
    "took 3 hours" means nothing without knowing what it ran on, and
    Kaggle doesn't always give you the accelerator you asked for."""
    import torch
    import ultralytics

    gpu_name, gpu_memory_gb = "cpu", None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )

    metadata = {
        "git_commit": _git_commit(),
        "hyperparameters": train_args,
        "hardware": {
            "gpu": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "ultralytics_version": ultralytics.__version__,
        "wall_clock_seconds": round(seconds, 1),
        "wall_clock_human": f"{seconds / 3600:.2f} h",
    }

    path = out_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nRun metadata written to {path}")
    print(f"  GPU:   {gpu_name} ({gpu_memory_gb} GB)")
    print(f"  Time:  {seconds / 3600:.2f} h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/doclaynet/doclaynet.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="rtdetr_doclaynet")
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny CPU run to prove the script works before spending GPU quota",
    )
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.batch, args.device, args.imgsz = 1, 2, "cpu", 320
        args.name = "smoke"

    from ultralytics import RTDETR
    from ultralytics.utils import SETTINGS

    # Ultralytics auto-registers a Ray Tune callback whenever `ray` is
    # importable, no check for whether Tune is actually running. Kaggle
    # ships ray pre-installed for other stuff, so this callback silently
    # activates and crashes end-of-epoch calling a Ray internal API that
    # doesn't exist in the installed version. Not using Ray Tune anywhere
    # here, so just turn the integration off before .train() runs.
    SETTINGS["raytune"] = False

    # Starting from COCO weights, not scratch. None of my 11 classes exist
    # in COCO so the detection head gets zero class knowledge for free -
    # the pretrained backbone just gives generic visual features. Training
    # from random init wouldn't converge in the epoch budget I have.
    model = RTDETR("rtdetr-l.pt")

    train_args = _build_train_args(args.data, args.epochs, args.batch, args.imgsz, args.device)
    train_args["name"] = args.name

    print(f"Training RT-DETR-L | {args.epochs} epochs | batch {args.batch} | {args.imgsz}px")
    started = time.time()
    results = model.train(**train_args)
    elapsed = time.time() - started

    out_dir = Path(results.save_dir)
    _write_run_metadata(out_dir, train_args, elapsed)
    print(f"\nBest weights: {out_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
