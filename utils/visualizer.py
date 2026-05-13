"""
visualizer.py
Drawing utilities: bounding boxes, labels, zoom crops, annotated frames.
"""
import cv2
import numpy as np
from PIL import Image
from core.preprocessor import zoom_crop, apply_filters

# Color palette per label
LABEL_COLORS = {
    "person":    (0, 200, 255),
    "car":       (255, 100, 50),
    "truck":     (180, 60, 220),
    "bus":       (200, 200, 0),
    "bicycle":   (0, 255, 150),
    "motorbike": (0, 180, 255),
    "bag":       (100, 255, 100),
    "backpack":  (80, 220, 80),
    "default":   (200, 200, 200),
    "unidentified": (120, 120, 120),
}


def get_color(label: str) -> tuple:
    return LABEL_COLORS.get(label.lower(), LABEL_COLORS["default"])


def draw_detections(frame: np.ndarray,
                    detections: list[dict],
                    highlight_labels: list[str] = None,
                    thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame.
    highlight_labels: if provided, these labels get a brighter/thicker box.
    """
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        label = det.get("display_label", det.get("label", "?"))
        conf  = det.get("confidence", 0)
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = get_color(label)

        is_highlight = (
            highlight_labels and
            any(h.lower() in label.lower() for h in highlight_labels)
        )
        lw = thickness + 2 if is_highlight else thickness

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)

        # Label background
        text   = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Highlight glow effect
        if is_highlight:
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)

    return out


def make_zoom_panel(frame: np.ndarray,
                    bbox: tuple,
                    filters: dict = None,
                    out_size: tuple = (280, 200),
                    label: str = "") -> np.ndarray:
    """Create a zoomed + enhanced crop panel for the matched region."""
    crop = zoom_crop(frame, bbox, out_size=out_size)

    if filters:
        crop = apply_filters(crop, filters)

    # Border
    cv2.rectangle(crop, (0, 0), (out_size[0]-1, out_size[1]-1), (0, 220, 255), 2)

    # Label overlay
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(crop, f"ZOOM: {label}", (6, 18), font, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    return crop


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert BGR numpy frame to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def annotate_match_frame(frame: np.ndarray,
                         match: dict,
                         filters: dict = None) -> dict:
    """
    Produce an annotated frame + zoom panel for a single match.
    Returns { annotated_frame (PIL), zoom_panel (PIL), timestamp_str }
    """
    # Draw box on frame
    annotated = draw_detections(
        frame,
        [{"display_label": match["label"], "confidence": match["confidence"], "bbox": match["bbox"]}],
        highlight_labels=[match["label"]],
    )

    # Zoom panel
    zoom = make_zoom_panel(
        frame,
        match["bbox"],
        filters=filters,
        label=f'{match["label"]} ({match["confidence"]:.0%})',
    )

    return {
        "annotated_frame": frame_to_pil(annotated),
        "zoom_panel":      frame_to_pil(zoom),
        "timestamp_str":   match["timestamp_str"],
    }


def build_timeline_image(matches: list[dict],
                          total_duration: float,
                          width: int = 700,
                          height: int = 40) -> np.ndarray:
    """
    Draw a timeline bar showing where matches occur in the video.
    """
    bar = np.full((height, width, 3), 20, dtype=np.uint8)
    # Background bar
    cv2.rectangle(bar, (0, height//2 - 4), (width-1, height//2 + 4), (50, 50, 60), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    seen_labels = {}

    for m in matches:
        if total_duration <= 0:
            continue
        x = int((m["timestamp_sec"] / total_duration) * width)
        x = max(2, min(x, width - 2))
        color = get_color(m.get("label", "default"))
        cv2.rectangle(bar, (x-2, 4), (x+2, height-4), color, -1)

    # Time labels
    for t_frac, label in [(0, "0s"), (0.5, f"{total_duration/2:.0f}s"),
                           (1.0, f"{total_duration:.0f}s")]:
        x = int(t_frac * (width - 1))
        cv2.putText(bar, label, (max(0, x-10), height-2), font, 0.3, (140, 140, 140), 1)

    return bar
