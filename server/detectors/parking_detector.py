"""
Illegal Parking Detector
Detects vehicles parked in restricted zones using YOLOv8.

Algorithm
─────────
1.  Detect all vehicle bounding boxes in the frame.
2.  Check whether each vehicle's bottom-centre point falls inside any
    pre-defined "no-parking" polygon for that specific camera.
3.  A vehicle that stays in a zone for >= DWELL_SECONDS is flagged as
    illegally parked.
4.  Once flagged, the event is not re-raised until the vehicle disappears
    and re-enters (simple cooldown).
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, Detection

log = logging.getLogger(__name__)

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}
CONFIDENCE_THRESHOLD = 0.55
DWELL_SECONDS        = 10    # seconds before raising an event
IOU_MATCH_THRESHOLD  = 0.35    # overlap to consider same vehicle across frames


@dataclass
class _VehicleTrack:
    """Internal tracking record for a vehicle in a no-park zone."""
    bbox:       List[int]
    first_seen: float = field(default_factory=time.monotonic)
    alerted:    bool  = False


def _iou(a: List[int], b: List[int]) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _point_in_polygon(px: int, py: int, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (px, py), False) >= 0


class IllegalParkingDetector(BaseDetector):
    name = "illegal_parking"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        dwell_seconds: float = DWELL_SECONDS,
    ):
        self._model        = YOLO(model_path)
        self._dwell        = dwell_seconds
        
        # Namespaced by camera_id to support multi-cam seamlessly
        self._zones: Dict[str, List[np.ndarray]] = {} 
        self._tracks: Dict[str, Dict[int, _VehicleTrack]] = {} 
        self._next_id: Dict[str, int] = {}

    # ─── Dynamic Zone Updater for UI ──────────────────────────────────────

    def update_zones(self, camera_id: str, zones: List[List[Tuple[int, int]]]):
        """Dynamically update the no-parking zones at runtime for a specific camera."""
        if not zones:
            self._zones.pop(camera_id, None)
            self._tracks.pop(camera_id, None)
            log.info("[%s] No-parking zones cleared via UI.", camera_id)
            return

        self._zones[camera_id] = [np.array(z, dtype=np.int32) for z in zones]
        
        # Clear existing tracks for this camera since the boundaries changed
        if camera_id in self._tracks:
            self._tracks[camera_id].clear() 
            
        log.info("[%s] No-parking zones updated dynamically via UI.", camera_id)

    # ─── Public API ───────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, camera_id: str = "unknown") -> List[Detection]:
        # Fast exit if this camera has no zones assigned
        cam_zones = self._zones.get(camera_id, [])
        if not cam_zones:
            return []

        # Initialize tracking memory for new cameras
        if camera_id not in self._tracks:
            self._tracks[camera_id] = {}
            self._next_id[camera_id] = 0

        results   = self._model(frame, verbose=False)[0]
        events: List[Detection] = []
        current_bboxes: List[List[int]] = []

        for box in results.boxes:
            label = results.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            if label.lower() not in VEHICLE_LABELS or conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = y2   # bottom-centre — ground contact point

            # Check against THIS camera's zones
            in_zone = any(
                _point_in_polygon(cx, cy, z) for z in cam_zones
            )
            if not in_zone:
                continue

            bbox = [x1, y1, x2, y2]
            current_bboxes.append(bbox)

            track_id = self._match_or_create(camera_id, bbox)
            track    = self._tracks[camera_id][track_id]
            dwell    = time.monotonic() - track.first_seen

            if dwell >= self._dwell and not track.alerted:
                track.alerted = True
                events.append(
                    Detection(
                        label="illegal_parking",
                        confidence=conf,
                        bbox=bbox,
                        camera_id=camera_id,
                        meta={
                            "vehicle_label": label,
                            "dwell_seconds": round(dwell, 1),
                            "detector":      self.name,
                            "track_id":      track_id,
                        },
                    )
                )
                log.info(
                    "[%s] Illegal parking — %s dwell=%.1fs bbox=%s",
                    camera_id, label, dwell, bbox,
                )

        # Prune stale tracks for THIS camera only
        self._prune_tracks(camera_id, current_bboxes)
        return events

    # ─── Zone rendering helper (for annotated preview) ───────────────────

    def draw_zones(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        cam_zones = self._zones.get(camera_id, [])
        if not cam_zones:
            return frame

        overlay = frame.copy()
        for zone in cam_zones:
            cv2.fillPoly(overlay, [zone], (0, 0, 200))
        return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    # ─── Internal helpers ────────────────────────────────────────────────

    def _match_or_create(self, camera_id: str, bbox: List[int]) -> int:
        best_id:  Optional[int] = None
        best_iou: float         = IOU_MATCH_THRESHOLD
        
        cam_tracks = self._tracks[camera_id]

        for tid, track in cam_tracks.items():
            score = _iou(bbox, track.bbox)
            if score > best_iou:
                best_iou = score
                best_id  = tid

        if best_id is not None:
            cam_tracks[best_id].bbox = bbox
            return best_id

        new_id = self._next_id[camera_id]
        cam_tracks[new_id] = _VehicleTrack(bbox=bbox)
        self._next_id[camera_id] += 1
        return new_id

    def _prune_tracks(self, camera_id: str, current_bboxes: List[List[int]]):
        cam_tracks = self._tracks[camera_id]
        stale = [
            tid for tid, track in cam_tracks.items()
            if not any(_iou(track.bbox, b) >= IOU_MATCH_THRESHOLD for b in current_bboxes)
        ]
        for tid in stale:
            del cam_tracks[tid]