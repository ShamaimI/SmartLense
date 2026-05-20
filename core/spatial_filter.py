"""
spatial_filter.py — Spatial Queries on Detections
Expects flat detection dicts (use flatten_detections() in app.py first).
"""
import numpy as np

REGIONS = {
    "top-left":     (0.0, 0.0, 0.5, 0.5),
    "top-right":    (0.5, 0.0, 1.0, 0.5),
    "bottom-left":  (0.0, 0.5, 0.5, 1.0),
    "bottom-right": (0.5, 0.5, 1.0, 1.0),
    "center":       (0.25, 0.25, 0.75, 0.75),
    "top":          (0.0, 0.0, 1.0, 0.4),
    "bottom":       (0.0, 0.6, 1.0, 1.0),
    "left":         (0.0, 0.0, 0.4, 1.0),
    "right":        (0.6, 0.0, 1.0, 1.0),
}


def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_in_region(bbox, frame_shape, region_name):
    if region_name not in REGIONS:
        return True
    h, w = frame_shape[:2]
    rx1, ry1, rx2, ry2 = REGIONS[region_name]
    cx, cy = get_bbox_center(bbox)
    return rx1 <= cx/w <= rx2 and ry1 <= cy/h <= ry2


def filter_by_region(flat_detections, region_name):
    if not region_name or region_name == "Any":
        return flat_detections
    return [
        det for det in flat_detections
        if is_in_region(det["bbox"], det["image"].shape, region_name)
    ]


def find_co_occurring_objects(flat_detections, label_a, label_b, max_distance_px=200):
    from collections import defaultdict
    by_timestamp = defaultdict(list)
    for det in flat_detections:
        by_timestamp[det["timestamp_sec"]].append(det)

    results = []
    for ts, dets in by_timestamp.items():
        a_dets = [d for d in dets if d["label"] == label_a]
        b_dets = [d for d in dets if d["label"] == label_b]
        for a in a_dets:
            for b in b_dets:
                cx1, cy1 = get_bbox_center(a["bbox"])
                cx2, cy2 = get_bbox_center(b["bbox"])
                dist = float(np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2))
                if dist <= max_distance_px:
                    results.append({
                        "timestamp_sec": ts,
                        "timestamp_str": a["timestamp_str"],
                        "object_a":      a,
                        "object_b":      b,
                        "distance_px":   round(dist, 1),
                        "image":         a["image"],
                        "label":         f"{label_a}+{label_b}",
                        "clip_similarity": 1.0,
                        "confidence":    min(a["confidence"], b["confidence"]),
                        "bbox":          a["bbox"],
                    })

    return sorted(results, key=lambda x: x["distance_px"])


def count_objects_in_region(flat_detections, region_name, label=None):
    filtered = filter_by_region(flat_detections, region_name)
    if label:
        filtered = [d for d in filtered if d["label"] == label]
    return len(filtered)