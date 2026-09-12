"""
Wraps the trained RT-DETR weights so the rest of the app never touches
Ultralytics directly.

I did this for a boring but real reason: I want to be able to write tests for
main.py and the reasoning pipeline without a GPU, without the weights file, and
without waiting on model.predict() to run. If Detector is the only thing that
imports ultralytics, everything else can be tested against a stub that returns
canned Detection objects in microseconds. The alternative - sprinkling
`from ultralytics import RTDETR` through the route handlers - would mean every
test of my API logic is secretly also a test of the ML library.

The other thing this file owns is deciding what "the model isn't ready yet"
looks like. I did not want model loading to happen at import time, because that
means the app fails to even start if the weights path is wrong, which makes a
health check useless - there is nothing left running to report the problem. So
loading is lazy and `is_loaded` can be asked about safely at any point.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.constants import DEFAULT_QUERY_BUDGET, ID_TO_CLASS
from app.schemas import BBox, Detection


class Detector:
    """Loads RT-DETR weights on first use and turns raw predictions into the
    Detection objects the rest of the app works with."""

    def __init__(self, weights_path: str | None = None):
        self._weights_path = weights_path or os.environ.get("MODEL_PATH", "./weights/best.pt")
        self._model = None
        self.query_budget = DEFAULT_QUERY_BUDGET

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """
        Loads the weights if they are not already loaded.

        Called once, lazily, from the first request or from the app's startup
        hook. I check the path exists myself before handing it to Ultralytics,
        because their own error for a missing file is a stack trace from deep
        inside torch.load, and "no such file" is a much faster thing for me to
        read at 1am than that traceback is.
        """
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
        """
        Runs inference on one image and returns the detections plus how long it
        took, in milliseconds.

        I return the timing from here rather than measuring it in the route
        handler, because "how long did the model take" and "how long did the
        whole HTTP request take" are different questions, and conflating them
        would blame network/serialisation overhead on the model.

        conf=0.25 as a default deliberately keeps the confidence guardrail's
        job separate from the detector's job. The detector's role is just to
        report what it saw, including things it is not very sure about; whether
        that uncertainty is good enough to answer a question is a decision the
        guardrail makes afterwards, on the full picture of all the detections,
        not something baked into a single detector-level cutoff.
        """
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
