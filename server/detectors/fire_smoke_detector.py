"""
Fire and Smoke Detector

Uses the general YOLO model to detect fire and smoke events.
"""

from __future__ import annotations

import os
import logging
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from .base import BaseDetector, Detection

log = logging.getLogger(__name__)

# Pull confidence threshold from .env, default to 0.45
CONFIDENCE_THRESHOLD = float(os.getenv("FIRE_SMOKE_CONFIDENCE", "0.25"))
TARGET_LABELS = {"fire", "smoke"}

class FireSmokeDetector(BaseDetector):
    name = "fire_smoke"

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.getenv("FIRE_MODEL", "fire.pt")
            
        log.info("[%s] Loading YOLO model: %s with %.2f confidence threshold", 
                 self.name, model_path, CONFIDENCE_THRESHOLD)

        try:
            self._model = YOLO(model_path)
            # --- ADD THIS LINE TO DEBUG CLASS NAMES ---
            log.info("Fire/Smoke Model loaded. Available classes: %s", self._model.names)
        except Exception:
            log.exception("Failed to load YOLO model for Fire/Smoke. Ensure %s exists.", model_path)
            raise

        if torch.cuda.is_available():
            self._model.to("cuda")
            self._device = 0
            log.info("FireSmokeDetector using GPU: %s", torch.cuda.get_device_name(0))
        else:
            self._device = "cpu"
            log.warning("FireSmokeDetector using CPU")

    def detect(self, frame: np.ndarray, camera_id: str = "unknown") -> List[Detection]:
        if frame is None:
            return []

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        try:
            results = self._model.predict(
                source=frame,
                device=self._device,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )[0]
        except Exception:
            log.exception("[%s] YOLO inference failed for fire/smoke", camera_id)
            return []

        detections: List[Detection] = []

        if results.boxes is None:
            return []

        for box in results.boxes:
            class_id = int(box.cls[0])
            label = results.names[class_id].lower()
            conf = float(box.conf[0])

            # Filter for only fire and smoke labels
            if label not in TARGET_LABELS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    camera_id=camera_id,
                    meta={
                        "detector": self.name,
                        "severity": "high",
                    },
                )
            )

        if detections:
            log.warning("[%s] ALERT: %d Fire/Smoke Detections Found!", camera_id, len(detections))

        return detections