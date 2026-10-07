"""
YOLOv8 person detector for broadcast-style player boxes (README / Kosolapov-style stack).
Falls back is handled in pipeline_worker if ultralytics or weights are unavailable.
"""

from __future__ import annotations

import numpy as np


class YoloPersonDetector:
    def __init__(self, device: str = "cpu"):
        from ultralytics import YOLO
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        custom = root / "models" / "weights" / "player_yolov8.pt"
        base = root / "models" / "weights" / "yolov8n.pt"
        if custom.is_file():
            weights = str(custom)
        elif base.is_file():
            weights = str(base)
        else:
            weights = "yolov8n.pt"
        self._device = 0 if device == "cuda" else "cpu"
        self.model = YOLO(weights)

    def detect(self, image: np.ndarray, person_min_score: float = 0.42):
        """
        image: BGR uint8 (OpenCV).
        Returns (list of xyxy numpy arrays, list of float scores): same contract as PersonDetector.detect.
        """
        res = self.model.predict(
            source=image,
            verbose=False,
            classes=[0],
            conf=float(person_min_score),
            device=self._device,
        )
        if not res:
            return [], []
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return [], []
        boxes = []
        scores = []
        for b in r0.boxes:
            xyxy = b.xyxy[0].detach().cpu().numpy().astype(np.float32)
            boxes.append(xyxy)
            scores.append(float(b.conf[0].detach().cpu()))
        return boxes, scores
