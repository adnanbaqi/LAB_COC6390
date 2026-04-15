# detectors/tripling_detector.py
from __future__ import annotations

import os
import logging
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from .base import BaseDetector, Detection

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.5"))

class TriplingDetector(BaseDetector):
    name = "tripling"

    def __init__(self, model_path: str = "yolov8n.pt"):
        log.info("Loading YOLO model for Tripling: %s", model_path)
        
        try:
            self._model = YOLO(model_path)
        except Exception:
            log.exception("Failed to load YOLO model for Tripling.")
            raise

        # Leverages your RTX 4060 for hardware acceleration
        if torch.cuda.is_available():
            self._model.to("cuda")
            self._device = 0
            log.info("TriplingDetector using GPU: %s", torch.cuda.get_device_name(0))
        else:
            self._device = "cpu"
            log.warning("TriplingDetector using CPU")

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
            log.exception("[%s] YOLO inference failed for tripling", camera_id)
            return []

        persons = []
        bikes = []

        if results.boxes is None:
            return []

        # 1. Categorize Detections
        for box in results.boxes:
            class_id = int(box.cls[0])
            label = results.names[class_id]
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if label == "person":
                persons.append({"bbox": (x1, y1, x2, y2), "conf": conf})
            elif label == "motorcycle":
                bikes.append({"bbox": (x1, y1, x2, y2), "conf": conf})

        detections: List[Detection] = []

        # 2. Check Overlap (Tripling Logic)
        for bike in bikes:
            bx1, by1, bx2, by2 = bike["bbox"]
            person_count = 0
            
            for person in persons:
                px1, py1, px2, py2 = person["bbox"]
                # Overlap logic
                if not (px2 < bx1 or px1 > bx2 or py2 < by1 or py1 > by2):
                    person_count += 1
            
            if person_count >= 3:
                detections.append(
                    Detection(
                        label="tripling",
                        confidence=bike["conf"],  # Using the motorcycle's confidence
                        bbox=[bx1, by1, bx2, by2], # Bounding box of the motorcycle
                        camera_id=camera_id,
                        meta={
                            "detector": self.name,
                            "person_count": person_count,
                            "severity": "high",
                        },
                    )
                )

        if detections:
            log.warning("[%s] ALERT: %d Tripling Events Found!", camera_id, len(detections))

        return detections