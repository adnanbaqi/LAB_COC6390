"""
Central Server — Flask REST API

Cameras are auto-registered on startup by reading from a .env file.
Manual registration via POST /api/cameras is also supported for runtime additions.

Routes
------
GET    /api/cameras                      List all cameras + live stats
POST   /api/cameras                      Manually register an extra camera
DELETE /api/cameras/<id>                 Remove a camera
GET    /api/cameras/<id>/feed            MJPEG annotated live stream
GET    /api/cameras/<id>/snapshot        Latest JPEG frame
GET    /api/parking/events               Parking log (filter: camera_id, resolved, limit)
POST   /api/parking/events/<id>/resolve  Mark a parking event as resolved
GET    /api/stats                        Detection counts by label
GET    /health                           Health check
"""

from __future__ import annotations

import os
import json
import logging
import time
import threading
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, abort

from .processor import ProcessorManager, StreamProcessor
from .detectors.trash_detector import TrashDetector
from .detectors.parking_detector import IllegalParkingDetector
from .detectors.fire_smoke_detector import FireSmokeDetector
from .detectors.tripling_detector import TriplingDetector
from .detectors.traffic_density_detector import TrafficDensityDetector

from .db.mongo import get_parking_events, resolve_parking_event, get_detection_stats, get_latest_traffic_data 

log = logging.getLogger(__name__)

# Load variables from the .env file
load_dotenv()

# ─── Dynamic Camera Loader ────────────────────────────────────────────────────

def load_cameras_from_env() -> list:
    """Scans environment variables for camera configurations."""
    cameras = []
    # FIX: Default zones should be EMPTY
    default_zones = [] 
    
    for i in range(1, 21):
        cam_id = os.getenv(f"CAMERA_{i}_ID")
        url = os.getenv(f"CAMERA_{i}_URL")
        zones_str = os.getenv(f"CAMERA_{i}_ZONES")
        
        if cam_id and url:
            # Use provided zones or an empty list
            zones = json.loads(zones_str) if zones_str else []
            
            cameras.append({
                "camera_id": cam_id,
                "stream_url": url,
                "parking_zones": zones
            })
    return cameras

PI_CAMERAS = load_cameras_from_env()


# ─── App + manager ────────────────────────────────────────────────────────────

app     = Flask(__name__)
manager = ProcessorManager()


# ─── Detector factory ─────────────────────────────────────────────────────────

def _build_detectors(camera_id: str, parking_zones: Optional[list] = None) -> list:
    """Build the detector stack for a camera."""
    # FIX: Remove the hardcoded [(0,320)...] logic
    zones = parking_zones or [] 
    
    parking_detector = IllegalParkingDetector()
    parking_detector.update_zones(camera_id, zones)

    return [
        TrashDetector(),
        parking_detector,
        FireSmokeDetector(),
        TriplingDetector(),
        TrafficDensityDetector(),
    ]


def _register(camera_id: str, stream_url: str, parking_zones: Optional[list] = None):
    """Create a StreamProcessor and add it to the manager."""
    proc = StreamProcessor(
        camera_id=camera_id,
        stream_url=stream_url,
        detectors=_build_detectors(camera_id, parking_zones),
    )
    manager.add(proc)
    log.info("Camera registered: %s -> %s", camera_id, stream_url)


# ─── Auto-register Cameras on import ──────────────────────────────────────────

if not PI_CAMERAS:
    log.warning("No cameras found in .env file! Server is running empty.")
else:
    log.info("Loaded %d cameras from environment.", len(PI_CAMERAS))

for _cam in PI_CAMERAS:
    _register(
        camera_id=    _cam["camera_id"],
        stream_url=   _cam["stream_url"],
        parking_zones=_cam.get("parking_zones"),
    )


# ─── Dynamic Stream Monitor ───────────────────────────────────────────────────

def _stream_health_monitor():
    """
    Background worker that checks stream health every 5 minutes.
    If a stream is disconnected for 3 consecutive checks, it drops the node.
    """
    RETRY_INTERVAL = 300  # 5 minutes in seconds
    MAX_RETRIES = 3
    failed_attempts = {}

    while True:
        time.sleep(RETRY_INTERVAL)
        
        cams = manager.all_stats()
        
        for cam in cams:
            cam_id = cam.get("camera_id")
            
            if not cam.get("connected", False):
                failed_attempts[cam_id] = failed_attempts.get(cam_id, 0) + 1
                log.warning("[%s] Stream unreachable. Attempt %d/%d", 
                            cam_id, failed_attempts[cam_id], MAX_RETRIES)
                
                # 3 Strikes: Remove the camera
                if failed_attempts[cam_id] >= MAX_RETRIES:
                    log.error("[%s] Failed %d times. Skipping and removing streaming node.", 
                              cam_id, MAX_RETRIES)
                    manager.remove(cam_id)
                    failed_attempts.pop(cam_id, None)
            else:
                if cam_id in failed_attempts:
                    log.info("[%s] Stream recovered.", cam_id)
                    failed_attempts.pop(cam_id, None)

# Start the monitor in a background daemon thread so it dies when the Flask app exits
monitor_thread = threading.Thread(target=_stream_health_monitor, daemon=True)
monitor_thread.start()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify(manager.all_stats())


@app.route("/api/cameras", methods=["POST"])
def register_camera():
    """Manually register an additional camera at runtime."""
    body      = request.get_json(force=True)
    camera_id = body.get("camera_id")
    url       = body.get("stream_url")
    zones     = body.get("parking_zones")

    if not camera_id or not url:
        abort(400, "camera_id and stream_url are required")
    if manager.get(camera_id):
        abort(409, f"Camera '{camera_id}' is already registered")

    _register(camera_id, url, zones)
    return jsonify({"status": "ok", "camera_id": camera_id}), 201


@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
def unregister_camera(camera_id):
    if not manager.get(camera_id):
        abort(404, f"Camera '{camera_id}' not found")
    manager.remove(camera_id)
    return jsonify({"status": "removed", "camera_id": camera_id})


# ─── Live feed + snapshot ─────────────────────────────────────────────────────

@app.route("/api/cameras/<camera_id>/feed")
def camera_feed(camera_id):
    proc = manager.get(camera_id)
    if proc is None:
        abort(404, f"Camera '{camera_id}' not found")

    def generate():
        timeout = 10
        start = time.monotonic()
        frame = None
        while frame is None and (time.monotonic() - start) < timeout:
            frame = proc.get_latest_frame()
            if frame is None:
                time.sleep(0.1)

        if frame is None:
            log.error("[%s] No frame available after %ds", camera_id, timeout)
            return

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        while True:
            frame = proc.get_latest_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/cameras/<camera_id>/snapshot")
def camera_snapshot(camera_id):
    proc = manager.get(camera_id)
    if proc is None:
        abort(404, f"Camera '{camera_id}' not found")
    frame = proc.get_latest_frame()
    if frame is None:
        abort(503, "No frame available yet — stream may still be connecting")
    return Response(frame, mimetype="image/jpeg")


# ─── Parking events ───────────────────────────────────────────────────────────

@app.route("/api/parking/events", methods=["GET"])
def list_parking():
    camera_id    = request.args.get("camera_id")
    resolved_str = request.args.get("resolved")
    limit        = int(request.args.get("limit", 100))

    resolved_bool: Optional[bool] = None
    if resolved_str is not None:
        resolved_bool = resolved_str.lower() == "true"

    events = get_parking_events(
        camera_id=camera_id,
        resolved=resolved_bool,
        limit=limit,
    )
    return jsonify(events)

@app.route('/api/traffic/latest')
def get_traffic_data():
    camera_id = request.args.get("camera_id") # Optional filter
    
    # Call the new helper function instead of the raw collection
    latest_data = get_latest_traffic_data(camera_id)
    
    if latest_data:
        latest_data['_id'] = str(latest_data['_id'])
        return jsonify(latest_data)
        
    return jsonify({"error": "No data available yet"})

@app.route("/api/parking/events/<event_id>/resolve", methods=["POST"])
def resolve_event(event_id):
    body    = request.get_json(force=True)
    officer = body.get("officer", "unknown")
    notes   = body.get("notes", "")

    ok = resolve_parking_event(event_id, officer=officer, notes=notes)
    if not ok:
        abort(404, "Event not found or already resolved")
    return jsonify({"status": "resolved", "event_id": event_id})


# ─── Stats + health ───────────────────────────────────────────────────────────
@app.route("/api/stats")
def stats():
    camera_id = request.args.get("camera_id")
    return jsonify(get_detection_stats(camera_id=camera_id))


@app.route("/health")
def health():
    cams = manager.all_stats()
    return jsonify({
        "status":        "ok",
        "cameras_total": len(cams),
        "cameras_live":  sum(1 for c in cams if c.get("connected", False)),
    })

@app.route("/api/cameras/<camera_id>/zones", methods=["POST"])
def update_camera_zones(camera_id):
    body = request.get_json(force=True)
    new_zones = body.get("zones")

    if new_zones is None or not isinstance(new_zones, list):
        abort(400, "Invalid zones format. Expected a list of polygons.")

    proc = manager.get(camera_id)
    if not proc:
        abort(404, f"Camera '{camera_id}' not found")

    # Handle clear — empty list is valid
    if not new_zones or not any(new_zones):
        for detector in proc.detectors:
            if detector.name == "illegal_parking":
                # ADDED camera_id
                detector.update_zones(camera_id, [])
        return jsonify({"status": "cleared", "camera_id": camera_id})

    # ── Scale from canvas space (640x480) to real frame size ──
    frame = proc.get_latest_frame()
    frame_w, frame_h = 640, 480  # fallback
    if frame is not None:
        import cv2, numpy as _np
        arr = cv2.imdecode(_np.frombuffer(frame, _np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            frame_h, frame_w = arr.shape[:2]

    scale_x = frame_w / 640
    scale_y = frame_h / 480

    scaled_zones = [
        [[int(x * scale_x), int(y * scale_y)] for x, y in polygon]
        for polygon in new_zones
    ]

    log.info("[%s] Zone update — frame=%dx%d scale=(%.2f,%.2f) zones=%s",
             camera_id, frame_w, frame_h, scale_x, scale_y, scaled_zones)

    updated = False
    for detector in proc.detectors:
        if detector.name == "illegal_parking":
            # ADDED camera_id
            detector.update_zones(camera_id, scaled_zones)
            updated = True

    if not updated:
        abort(400, "Illegal parking detector not active on this camera.")

    return jsonify({
        "status":       "success",
        "camera_id":    camera_id,
        "frame_size":   [frame_w, frame_h],
        "scaled_zones": scaled_zones,
    })

# ─── Cleanup on shutdown ──────────────────────────────────────────────────────

import atexit
atexit.register(manager.stop_all)