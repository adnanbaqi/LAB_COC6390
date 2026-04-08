"""
Stream Processor
────────────────
Connects to a Pi MJPEG stream or YouTube Live Stream, pulls frames in a background thread,
runs all registered detectors, logs events, and exposes the latest
annotated frame as a JPEG byte-string for the Flask API to serve.
"""

from __future__ import annotations

import os
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import yt_dlp
from dotenv import load_dotenv

from .detectors.base import BaseDetector, Detection
from .db.mongo import log_detection, log_parking_event
from .utils.snapshot import save_snapshot

from .utils.alpr import extract_license_plate
from .utils.notifier import send_sms_alert, send_rto_email

load_dotenv()
log = logging.getLogger(__name__)

READ_TIMEOUT   = 5
RETRY_DELAY    = 3
FRAME_RESIZE   = (640, 480)

# 5 Minute Throttling for high-priority notifications (SMS/Email)
NOTIFICATION_COOLDOWN_SECONDS = 60 

DETECTION_INTERVAL_SECONDS = float(os.getenv("DETECTION_INTERVAL_SECONDS", "3.0"))


@dataclass
class CameraStats:
    camera_id:     str
    stream_url:    str
    frames_read:   int = 0
    detections:    int = 0
    errors:        int = 0
    fps:           float = 0.0
    connected:     bool = False
    last_frame_ts: float = field(default_factory=time.monotonic)


class StreamProcessor:
    def __init__(
        self,
        camera_id:  str,
        stream_url: str,
        detectors:  Optional[List[BaseDetector]] = None,
        save_snapshots: bool = True,
    ):
        self.camera_id      = camera_id
        self.stream_url     = stream_url
        self.detectors      = detectors or []
        self.save_snapshots = save_snapshots

        self.stats     = CameraStats(camera_id=camera_id, stream_url=stream_url)
        self._lock     = threading.Lock()
        self._latest   : Optional[bytes] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._running  = False
        self._thread   : Optional[threading.Thread] = None

        self._last_detections: List[Detection] = []
        self._last_detection_time = 0.0

        # Namespaced cooldowns for notifications
        # key: notification_type (e.g. "fire" or "parking") -> last_timestamp
        self._notification_locks: Dict[str, float] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("[%s] Processor started -> %s", self.camera_id, self.stream_url)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def get_latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest

    def get_stats(self) -> dict:
        s = self.stats
        active_zones = []
        for detector in self.detectors:
            if detector.name == "illegal_parking" and hasattr(detector, "_zones"):
                # FIXED: Correctly access the namespaced dictionary via camera_id
                cam_zones = detector._zones.get(self.camera_id, [])
                active_zones = [z.tolist() for z in cam_zones]
                break

        return {
            "camera_id":   s.camera_id,
            "stream_url":  s.stream_url,
            "frames_read": s.frames_read,
            "detections":  s.detections,
            "errors":      s.errors,
            "fps":         round(s.fps, 2),
            "connected":   s.connected,
            "zones":       active_zones,
        }

    def _loop(self):
        while self._running:
            cap = self._open_stream()
            if cap is None:
                time.sleep(RETRY_DELAY)
                continue

            self.stats.connected = True
            fps_timer  = time.monotonic()
            fps_frames = 0

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    log.warning("[%s] Frame read failed", self.camera_id)
                    self.stats.errors += 1
                    break

                frame = cv2.resize(frame, FRAME_RESIZE)
                self.stats.frames_read += 1
                fps_frames += 1

                if fps_frames >= 30:
                    elapsed = time.monotonic() - fps_timer
                    self.stats.fps = fps_frames / elapsed if elapsed else 0
                    fps_frames = 0
                    fps_timer  = time.monotonic()

                current_time = time.monotonic()
                if current_time - self._last_detection_time >= DETECTION_INTERVAL_SECONDS:
                    detections = self._run_detectors(frame)
                    self._last_detections = detections
                    self._last_detection_time = current_time
                else:
                    detections = self._last_detections

                annotated = self._draw_detections(frame, detections)

                if current_time - self._last_detection_time < 0.1:
                    self.stats.detections += len(detections)

                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self._lock:
                        self._latest = buf.tobytes()

            cap.release()
            self.stats.connected = False
            time.sleep(RETRY_DELAY)

    def _open_stream(self) -> Optional[cv2.VideoCapture]:
        target_url = self.stream_url
        if "youtube.com" in target_url or "youtu.be" in target_url:
            try:
                ydl_opts = {'format': 'best', 'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(target_url, download=False)
                    target_url = info.get('url', target_url)
            except Exception as e:
                log.error("[%s] YouTube resolver failed: %s", self.camera_id, e)
                return None

        cap = cv2.VideoCapture(target_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        return cap

    def _run_detectors(self, frame: np.ndarray) -> List[Detection]:
        all_events: List[Detection] = []
        for detector in self.detectors:
            try:
                events = detector.detect(frame, camera_id=self.camera_id)
                for det in events:
                    all_events.append(det)
                    self._persist(frame, det)
            except Exception:
                log.exception("[%s] Detector %s failed", self.camera_id, detector.name)
        return all_events

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated = frame.copy()
        # Draw the active parking zones on the stream
        for detector in self.detectors:
            if detector.name == "illegal_parking" and hasattr(detector, "draw_zones"):
                annotated = detector.draw_zones(annotated, self.camera_id)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if det.label in ["fire", "smoke"]:
                color = (0, 165, 255)
            elif det.label == "illegal_parking":
                color = (0, 0, 255)
            else:
                color = (0, 200, 50)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{det.label} {det.confidence:.0%}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return annotated

    def _persist(self, raw_frame: np.ndarray, det: Detection):
        """Save snapshot and log event with strict notification throttling."""
        now = time.time()
        
        # Use a label-based key for throttling (e.g. "fire" or "illegal_parking")
        notif_type = "parking" if det.label == "illegal_parking" else det.label
        last_sent = self._notification_locks.get(notif_type, 0)

        # 1. Logic for Database Logging (No cooldown for DB logs)
        snapshot_path: Optional[str] = None
        if self.save_snapshots:
            snapshot_path = save_snapshot(raw_frame, det.label, self.camera_id, det.bbox)

        if det.label == "illegal_parking":
            log_parking_event(det, snapshot_path)
        else:
            log_detection(det, snapshot_path)

        # 2. Logic for high-priority notifications (SMS/Email) with 5-min cooldown
        if now - last_sent < NOTIFICATION_COOLDOWN_SECONDS:
            return # Skip notification, already sent recently

        self._notification_locks[notif_type] = now

        try:
            if det.label == "illegal_parking":
                plate_number = extract_license_plate(raw_frame, det.bbox)
                det.meta["license_plate"] = plate_number
                
                send_sms_alert(f"PARKING ALERT: Cam {self.camera_id}\nPlate: {plate_number}")
                send_rto_email(plate_number, self.camera_id, det.meta.get('dwell_seconds', 0), snapshot_path)
            
            elif det.label in ["fire", "smoke"]:
                msg = f"CRITICAL: {det.label.upper()} DETECTED!\nCam: {self.camera_id}"
                send_sms_alert(msg)
                # Reuse RTO email logic for fire emergency
                send_rto_email("FIRE_EMERGENCY", self.camera_id, 0.0, snapshot_path)

        except Exception:
            log.exception("Notification pipeline failed")


class ProcessorManager:
    def __init__(self):
        self._processors: Dict[str, StreamProcessor] = {}

    def add(self, processor: StreamProcessor):
        self._processors[processor.camera_id] = processor
        processor.start()

    def remove(self, camera_id: str):
        proc = self._processors.pop(camera_id, None)
        if proc:
            proc.stop()

    def get(self, camera_id: str) -> Optional[StreamProcessor]:
        return self._processors.get(camera_id)

    def all_stats(self) -> List[dict]:
        return [p.get_stats() for p in self._processors.values()]

    def stop_all(self):
        for proc in self._processors.values():
            proc.stop()
        self._processors.clear()