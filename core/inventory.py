"""
inventory.py
Scans the full video and builds a complete object inventory.
Flags low-confidence detections as Unidentified.
"""
import numpy as np
from collections import defaultdict


def build_inventory(detected_frames: list[dict]) -> dict:
    """
    Scan all detected frames and build a complete object inventory.

    Returns:
      {
        "objects": { label: { count, avg_conf, first_seen, last_seen, frames } },
        "unidentified": [ { timestamp_str, confidence, bbox, frame_idx } ],
        "summary": { total_detections, unique_objects, unidentified_count }
      }
    """
    objects     = defaultdict(lambda: {
        "count": 0, "confidences": [], "first_seen": None, "last_seen": None,
        "frames": [], "thumbnails": []
    })
    unidentified = []

    for frame_data in detected_frames:
        for det in frame_data.get("detections", []):
            if not det["identified"]:
                unidentified.append({
                    "timestamp_str": frame_data["timestamp_str"],
                    "timestamp_sec": frame_data["timestamp_sec"],
                    "confidence":    det["confidence"],
                    "bbox":          det["bbox"],
                    "frame_idx":     frame_data["frame_idx"],
                    "image":         frame_data["image"],
                })
                continue

            label = det["label"]
            obj   = objects[label]
            obj["count"]       += 1
            obj["confidences"].append(det["confidence"])
            obj["frames"].append(frame_data["timestamp_sec"])

            if obj["first_seen"] is None:
                obj["first_seen"]    = frame_data["timestamp_str"]
                obj["first_seen_sec"] = frame_data["timestamp_sec"]
                obj["thumbnails"].append({"image": frame_data["image"], "bbox": det["bbox"]})

            obj["last_seen"]    = frame_data["timestamp_str"]
            obj["last_seen_sec"] = frame_data["timestamp_sec"]

    # Clean up: compute averages, deduplicate nearby timestamps
    inventory_out = {}
    for label, data in objects.items():
        deduped_times = _deduplicate_times(data["frames"], gap=1.0)
        inventory_out[label] = {
            "count":         data["count"],
            "avg_confidence": round(np.mean(data["confidences"]), 3),
            "first_seen":    data["first_seen"],
            "last_seen":     data["last_seen"],
            "appearance_times": [_sec_to_str(t) for t in deduped_times],
            "thumbnail":     data["thumbnails"][0] if data["thumbnails"] else None,
        }

    # Sort by count desc
    inventory_out = dict(
        sorted(inventory_out.items(), key=lambda x: x[1]["count"], reverse=True)
    )

    return {
        "objects":     inventory_out,
        "unidentified": unidentified,
        "summary": {
            "total_detections":  sum(d["count"] for d in inventory_out.values()),
            "unique_objects":    len(inventory_out),
            "unidentified_count": len(unidentified),
        },
    }


def _deduplicate_times(times: list[float], gap: float = 1.0) -> list[float]:
    """Remove timestamps that are too close together."""
    if not times:
        return []
    out = [times[0]]
    for t in sorted(times[1:]):
        if t - out[-1] >= gap:
            out.append(t)
    return out


def _sec_to_str(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"
