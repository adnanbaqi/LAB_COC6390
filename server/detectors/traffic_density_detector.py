# detectors/traffic_density_detector.py
from __future__ import annotations

import time
import logging
from datetime import datetime
from typing import List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .base import BaseDetector, Detection
from ..db.mongo import log_traffic_density  # <-- Import the new helper

log = logging.getLogger(__name__)

class TrafficDensityDetector(BaseDetector):
    name = "traffic_density"

    def __init__(self, model_path: str = "yolov8n.pt", save_interval: int = 1):
        log.info("Loading YOLO model for Traffic Density: %s", model_path)
        self._model = YOLO(model_path)
        
        if torch.cuda.is_available():
            self._model.to("cuda")
            self._device = 0
        else:
            self._device = "cpu"

        self.target_classes = {"car", "motorcycle", "bus", "truck"}
        self.save_interval = save_interval
        self.last_save_time = time.time()

    def detect(self, frame: np.ndarray, camera_id: str = "unknown") -> List[Detection]:
        if frame is None:
            return []

        # Run inference
        results = self._model.predict(source=frame, device=self._device, conf=0.4, verbose=False)[0]
        
        class_counts = {"car": 0, "bike": 0, "bus": 0, "truck": 0}
        detections: List[Detection] = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                
                if label in self.target_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Map 'motorcycle' to 'bike' to match your schema
                    count_key = "bike" if label == "motorcycle" else label
                    class_counts[count_key] += 1
                    
                    # Prefix label with "vehicle_" to prevent DB/Snapshot spam in the main processor
                    detections.append(Detection(
                        label=f"vehicle_{label}",
                        confidence=conf,
                        bbox=[x1, y1, x2, y2],
                        camera_id=camera_id,
                        meta={"detector": self.name}
                    ))

        total_vehicles = sum(class_counts.values())

        # Determine Density
        if total_vehicles < 5:
            density = "LOW"
        elif total_vehicles < 15:
            density = "MEDIUM"
        else:
            density = "HIGH"

        # MongoDB Save Logic (Executes every `save_interval` seconds)
        current_time = time.time()
        if (current_time - self.last_save_time) >= self.save_interval:
            data_entry = {
                "camera_id": camera_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "class_counts": class_counts,
                "total_vehicles": total_vehicles,
                "density": density
            }
            # Use the central mongo helper instead of a local client
            log_traffic_density(data_entry)
            self.last_save_time = current_time

        # Draw HUD directly on the frame
        y_offset = 30
        for key, val in class_counts.items():
            cv2.putText(frame, f"{key.capitalize()}: {val}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25

        y_offset += 10
        cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset += 35
        density_color = (0, 255, 0) if density == "LOW" else (0, 165, 255) if density == "MEDIUM" else (0, 0, 255)
        cv2.putText(frame, f"Density: {density}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2)

        return detections