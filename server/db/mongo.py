"""
MongoDB persistence layer.

Collections
───────────
  detections   — every raw detection event
  parking_logs — enriched illegal-parking events with snapshot path
  traffic_data — periodic traffic density and vehicle counts
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

from ..detectors.base import Detection

log = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ─── Connection ──────────────────────────────────────────────────────────────

_client: Optional[MongoClient] = None
_db     = None

# Fetch strictly from .env
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB", "traffic_db")

# Fail fast if the URI is missing
if not MONGO_URI:
    raise ValueError("CRITICAL: MONGO_URI is not set in the environment variables. Please check your .env file.")



def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(
            MONGO_URI, 
            serverSelectionTimeoutMS=5_000,
            tlsCAFile=certifi.where()
        )
        _db     = _client[DB_NAME]
        _ensure_indexes()
        log.info("MongoDB connected → [MASKED_URI] / %s", DB_NAME)
    return _db


def _ensure_indexes():
    db = _db
    db.detections.create_index([("timestamp", DESCENDING)])
    db.detections.create_index([("label", 1)])
    db.detections.create_index([("camera_id", 1)])
    
    db.parking_logs.create_index([("timestamp", DESCENDING)])
    db.parking_logs.create_index([("camera_id", 1)])
    db.parking_logs.create_index([("resolved", 1)])
    
    # New indexes for traffic data
    db.traffic_data.create_index([("timestamp", DESCENDING)])
    db.traffic_data.create_index([("camera_id", 1)])
    log.debug("Indexes ensured.")


# ─── Public helpers ──────────────────────────────────────────────────────────

def log_detection(detection: Detection, snapshot_path: Optional[str] = None):
    doc = detection.to_dict()
    if snapshot_path:
        doc["snapshot"] = snapshot_path
    get_db().detections.insert_one(doc)


def log_parking_event(detection: Detection, snapshot_path: Optional[str] = None):
    doc = {
        **detection.to_dict(),
        "snapshot":   snapshot_path,
        "resolved":   False,
        "resolved_at": None,
        "officer":    None,
        "notes":      None,
    }
    result = get_db().parking_logs.insert_one(doc)
    log.info("Parking event logged → _id=%s camera=%s", result.inserted_id, detection.camera_id)
    return str(result.inserted_id)


def resolve_parking_event(event_id: str, officer: str, notes: str = "") -> bool:
    from bson import ObjectId
    db     = get_db()
    result = db.parking_logs.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": {
            "resolved":    True,
            "resolved_at": datetime.utcnow().isoformat(),
            "officer":     officer,
            "notes":       notes,
        }},
    )
    return result.modified_count == 1


def get_parking_events(
    camera_id: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 100,
) -> list:
    query: dict = {}
    if camera_id is not None:
        query["camera_id"] = camera_id
    if resolved is not None:
        query["resolved"] = resolved

    cursor = (
        get_db()
        .parking_logs
        .find(query, {"_id": 0})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    return list(cursor)


def get_detection_stats(camera_id: Optional[str] = None) -> dict:
    db = get_db()
    
    pipeline = []
    match_query = {}
    
    if camera_id:
        match_query["camera_id"] = camera_id
        pipeline.append({"$match": match_query})
        
    pipeline.extend([
        {"$group": {"_id": "$label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ])
    
    labels = {r["_id"]: r["count"] for r in db.detections.aggregate(pipeline)}
    total  = db.detections.count_documents(match_query)
    
    return {"total": total, "by_label": labels}

# New Helper for Traffic Density
def log_traffic_density(data_entry: dict):
    """Insert periodic traffic density and vehicle count data."""
    try:
        get_db().traffic_data.insert_one(data_entry)
    except Exception as e:
        log.error("Failed to insert traffic data to MongoDB: %s", e)

def get_latest_traffic_data(camera_id: Optional[str] = None) -> Optional[dict]:
    """Fetch the most recent traffic density reading."""
    db = get_db()
    query = {"camera_id": camera_id} if camera_id else {}
    
    # Sort by timestamp descending to get the newest record
    from pymongo import DESCENDING
    return db.traffic_data.find_one(query, sort=[('timestamp', DESCENDING)])