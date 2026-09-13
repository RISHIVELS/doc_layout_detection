# Thin wrapper around Ultralytics so nothing else in the app imports it
# directly - lets me stub this out in tests without a GPU or real weights.
from __future__ import annotations

import os
import time
from pathlib import Path

from app.constants import DEFAULT_QUERY_BUDGET, ID_TO_CLASS
from app.schemas import BBox, Detection


class Detector:
    """Loads RT-DETR on first use, converts raw output to Detection objects."""

    def __init__(self, weights_path: str | None = None):
        self._weights_path = weights_path or os.environ.get("MODEL_PATH", "./weights/best.pt")
        self._model = None
        self.query_budget = DEFAULT_QUERY_BUDGET

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Lazy load - keeps a bad MODEL_PATH from crashing the whole app
        at startup. Checks the file exists first so the error is readable
        instead of a torch.load stack trace."""
        if self.is_loaded:
            return

        if not Path(self._weights_path).exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self._weights_path}'. "
                "Set MODEL_PATH in your .env, or see the README for the "
                "weights download link."
            )

        from ultralytics import RTDETR

        self._model = RTDETR(self._weights_path)

    def predict(self, image, conf: float = 0.25) -> tuple[list[Detection], float]:
        """Runs inference, returns (detections, inference_ms). Timing is
        measured here, not in the route handler, so it's not conflated with
        request/network overhead.

        conf=0.25 is intentionally permissive - the guardrail decides what
        counts as confident enough, not this default."""
        if not self.is_loaded:
            self.load()

        started = time.perf_counter()
        results = self._model.predict(image, conf=conf, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        detections = [
            Detection(
                class_name=ID_TO_CLASS.get(int(class_id), f"unknown_{int(class_id)}"),
                class_id=int(class_id),
                confidence=float(confidence),
                bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            )
            for class_id, confidence, (x1, y1, x2, y2) in zip(
                results.boxes.cls.tolist(),
                results.boxes.conf.tolist(),
                results.boxes.xyxy.tolist(),
            )
        ]

        return detections, elapsed_ms
