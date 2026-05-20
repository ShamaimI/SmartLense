"""
detector.py
Dual-mode object detection:
  - PRIMARY:  YOLOv8n (used automatically when weights are available locally)
  - FALLBACK: HOG people detector + background subtraction + color/shape analysis
              (runs 100% offline, no internet required)

When students run this on their own machine, YOLOv8 auto-downloads on first run.
"""
import numpy as np
import cv2

UNIDENTIFIED_THRESHOLD = 0.35

# ─────────────────────────────────────────────────────────────────────────────
#  YOLO (primary - requires yolov8n.pt downloaded)
# ─────────────────────────────────────────────────────────────────────────────
_yolo_model = None
_yolo_available = None          # None = not yet checked


def _try_load_yolo(model_name: str = "yolov8n.pt"):
    global _yolo_model, _yolo_available
    if _yolo_available is not None:
        return _yolo_available
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_name)
        _yolo_available = True
        print("[detector] YOLOv8 loaded successfully.")
    except Exception as e:
        _yolo_available = False
        print(f"[detector] YOLOv8 not available ({e}). Using offline fallback detector.")
    return _yolo_available


def _detect_yolo(frame: np.ndarray, conf_threshold: float,
                 category_filter: list) -> list[dict]:
    results = _yolo_model(frame, verbose=False, conf=conf_threshold)[0]
    detections = []
    for box in results.boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = _yolo_model.names[cls]
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        if category_filter and not any(label.lower() == c.lower() for c in category_filter):
            continue
        identified = conf >= UNIDENTIFIED_THRESHOLD
        detections.append({
            "label":         label,
            "confidence":    round(conf, 3),
            "bbox":          (x1, y1, x2, y2),
            "identified":    identified,
            "display_label": label if identified else "Unidentified",
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
#  OFFLINE FALLBACK DETECTOR
#  Uses: HOG people, background subtraction + contours, color histograms
# ─────────────────────────────────────────────────────────────────────────────

# Persistent background subtractor across frames
_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
_hog = None

# Color ranges for vehicle/object identification (HSV)
COLOR_LABELS = {
    "red vehicle":    ([0, 120, 70],   [10, 255, 255]),
    "blue vehicle":   ([100, 80, 50],  [130, 255, 255]),
    "yellow object":  ([20, 100, 100], [35, 255, 255]),
    "green object":   ([40, 60, 60],   [85, 255, 255]),
}

# Aspect-ratio heuristics for shape-based labelling
def _aspect_label(w: int, h: int) -> tuple[str, float]:
    """Guess label from bounding-box aspect ratio."""
    if w == 0 or h == 0:
        return "object", 0.30
    ratio = w / h
    area  = w * h
    if 0.3 < ratio < 0.7 and h > 60:
        return "person",  0.72
    elif ratio > 1.8 and area > 3000:
        return "car",     0.65
    elif ratio > 2.5 and area > 6000:
        return "truck",   0.60
    elif 0.8 < ratio < 1.8 and area < 3000:
        return "bag",     0.52
    elif 0.5 < ratio < 1.2 and area < 2500:
        return "bicycle", 0.55
    else:
        return "object",  0.28


def _get_hog():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog


def _detect_fallback(frame: np.ndarray, conf_threshold: float,
                     category_filter: list) -> list[dict]:
    """
    Offline detector combining:
      1. HOG sliding-window for people
      2. Background subtraction + contours for moving objects
      3. Aspect-ratio heuristics for labelling
    """
    h, w = frame.shape[:2]
    detections = []
    seen_boxes  = []   # for NMS

    # ── 1. HOG person detection ───────────────────────────────────────────
    hog = _get_hog()
    small = cv2.resize(frame, (320, 240))
    sx, sy = w / 320, h / 240
    rects, weights = hog.detectMultiScale(
        small, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    for i, (rx, ry, rw, rh) in enumerate(rects):
        conf = float(min(0.95, 0.55 + weights[i][0] * 0.1)) if len(weights) > i else 0.62
        if conf < conf_threshold:
            continue
        x1, y1 = int(rx * sx), int(ry * sy)
        x2, y2 = int((rx + rw) * sx), int((ry + rh) * sy)
        if _is_duplicate(x1, y1, x2, y2, seen_boxes):
            continue
        seen_boxes.append((x1, y1, x2, y2))
        if category_filter and "person" not in [c.lower() for c in category_filter]:
            continue
        detections.append({
            "label": "person", "confidence": round(conf, 3),
            "bbox": (x1, y1, x2, y2), "identified": conf >= UNIDENTIFIED_THRESHOLD,
            "display_label": "person",
        })

    # ── 2. Background subtraction + contours (moving objects) ────────────
    fg_mask = _bg_subtractor.apply(frame)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:          # ignore tiny blobs
            continue
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if _is_duplicate(rx, ry, rx+rw, ry+rh, seen_boxes, iou_thresh=0.4):
            continue
        seen_boxes.append((rx, ry, rx+rw, ry+rh))

        label, conf = _aspect_label(rw, rh)
        if conf < conf_threshold:
            label, conf = "object", conf_threshold   # promote to threshold

        # Refine label with color hint
        roi = frame[ry:ry+rh, rx:rx+rw]
        color_label = _color_hint(roi)
        if color_label:
            label = color_label
            conf  = max(conf, 0.58)

        if category_filter and label not in [c.lower() for c in category_filter]:
            continue

        identified = conf >= UNIDENTIFIED_THRESHOLD
        detections.append({
            "label": label, "confidence": round(conf, 3),
            "bbox": (rx, ry, rx+rw, ry+rh),
            "identified": identified,
            "display_label": label if identified else "Unidentified",
        })

    return detections


def _color_hint(roi: np.ndarray) -> str | None:
    """Return a color-based label hint if dominant color matches a known category."""
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for label, (lower, upper) in COLOR_LABELS.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > roi.size * 0.15:
            return label
    return None


def _is_duplicate(x1, y1, x2, y2, seen: list, iou_thresh: float = 0.5) -> bool:
    for sx1, sy1, sx2, sy2 in seen:
        inter_x1, inter_y1 = max(x1, sx1), max(y1, sy1)
        inter_x2, inter_y2 = min(x2, sx2), min(y2, sy2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = ((x2-x1)*(y2-y1) + (sx2-sx1)*(sy2-sy1) - inter_area)
        if union_area > 0 and inter_area / union_area > iou_thresh:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def detect_frame(frame: np.ndarray,
                 conf_threshold: float = 0.3,
                 category_filter: list[str] = None,
                 model_name: str = "yolov8n.pt") -> list[dict]:
    """
    Run detection on a single frame.
    Uses YOLOv8 if available, otherwise falls back to HOG + contour detector.
    """
    if _try_load_yolo(model_name):
        return _detect_yolo(frame, conf_threshold, category_filter)
    return _detect_fallback(frame, conf_threshold, category_filter)


def detect_all_frames(frames: list[dict],
                      conf_threshold: float = 0.3,
                      category_filter: list[str] = None,
                      model_name: str = "yolov8n.pt") -> list[dict]:
    """
    Run detection on all frame dicts returned by video_loader.
    Returns same list with 'detections' key added to each entry.
    """
    out = []
    for frame_data in frames:
        dets = detect_frame(
            frame_data["image"],
            conf_threshold=conf_threshold,
            category_filter=category_filter,
            model_name=model_name,
        )
        out.append({**frame_data, "detections": dets})
    return out
