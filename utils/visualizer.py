"""
visualizer.py — Bounding box annotation, zoom panels, timeline images.
"""
import cv2
import numpy as np
from PIL import Image


LABEL_COLORS = {
    "person":    (0, 229, 255),
    "car":       (0, 255, 128),
    "truck":     (255, 165, 0),
    "bus":       (255, 100, 0),
    "bicycle":   (180, 0, 255),
    "motorbike": (200, 0, 200),
    "bag":       (255, 255, 0),
    "backpack":  (200, 220, 0),
}
DEFAULT_COLOR = (100, 180, 255)


def annotate_match_frame(frame: np.ndarray, match: dict,
                          filters: dict = None) -> dict:
    """
    Draw bounding box on best-match frame and create a zoom panel.
    Returns dict with 'annotated_frame' (RGB PIL) and 'zoom_panel' (RGB PIL).
    """
    from core.preprocessor import apply_filters, zoom_crop

    annotated = frame.copy()
    bbox = match.get("bbox")
    label = match.get("label", "?")
    conf  = match.get("confidence", 0)
    sim   = match.get("clip_similarity", 0)

    color = LABEL_COLORS.get(label.lower(), DEFAULT_COLOR)

    if bbox:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f} | CLIP:{sim:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        zoom = zoom_crop(frame, bbox, zoom=2.0, out_size=(320, 240))
        if filters:
            zoom = apply_filters(zoom, filters)
    else:
        zoom = cv2.resize(frame, (320, 240))

    if filters:
        annotated = apply_filters(annotated, filters)

    return {
        "annotated_frame": Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)),
        "zoom_panel":      Image.fromarray(cv2.cvtColor(zoom,      cv2.COLOR_BGR2RGB)),
    }


def build_timeline_image(matches: list, total_duration_sec: float,
                          width: int = 800, height: int = 40) -> Image.Image:
    """
    Build a horizontal timeline image showing match timestamps.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (13, 18, 25)    # dark background

    # Track bar
    cv2.rectangle(img, (0, height//2 - 2), (width, height//2 + 2), (30, 50, 70), -1)

    if total_duration_sec <= 0:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for m in matches:
        ts  = m.get("timestamp_sec", 0)
        sim = m.get("clip_similarity", 0.5)
        x   = int((ts / total_duration_sec) * width)
        x   = max(1, min(width - 1, x))
        color_intensity = int(100 + sim * 155)
        color = (0, color_intensity, 255)
        cv2.line(img, (x, 4), (x, height - 4), color, 2)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def draw_trajectories(frame: np.ndarray, trajectories: dict,
                       max_points: int = 50) -> np.ndarray:
    """
    Draw trajectory paths on a frame for object tracking visualization.
    trajectories: { track_id: [(x, y, ts), ...] }
    """
    TRAJ_COLORS = [
        (0, 229, 255), (0, 255, 128), (255, 165, 0),
        (180, 0, 255), (255, 255, 0), (0, 180, 255),
        (255, 100, 0), (100, 255, 100),
    ]
    out = frame.copy()
    for i, (tid, points) in enumerate(trajectories.items()):
        color = TRAJ_COLORS[i % len(TRAJ_COLORS)]
        pts = [(int(p[0]), int(p[1])) for p in points[-max_points:]]
        for j in range(1, len(pts)):
            alpha = j / len(pts)
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(out, pts[j-1], pts[j], c, 2)
        if pts:
            cv2.circle(out, pts[-1], 5, color, -1)
            cv2.putText(out, f"#{tid}", (pts[-1][0]+6, pts[-1][1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def make_reid_mosaic(person_tracks: dict, max_persons: int = 6,
                     thumb_size: tuple = (80, 160)) -> Image.Image | None:
    """
    Build a side-by-side mosaic of person thumbnails from Re-ID results.
    """
    import cv2
    persons = list(person_tracks.items())[:max_persons]
    if not persons:
        return None

    thumbs = []
    for pid, dets in persons:
        if not dets:
            continue
        det = dets[0]
        frame = det.get("image")
        bbox  = det.get("bbox")
        if frame is None or bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if crop.size == 0:
            continue
        thumb = cv2.resize(crop, thumb_size)
        # Label
        cv2.rectangle(thumb, (0, 0), (thumb_size[0], 20), (0, 80, 120), -1)
        cv2.putText(thumb, f"Person #{pid}", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 229, 255), 1)
        thumbs.append(thumb)

    if not thumbs:
        return None

    mosaic = np.hstack(thumbs)
    return Image.fromarray(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
