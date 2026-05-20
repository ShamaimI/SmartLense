"""
live_camera.py — Live Webcam / RTSP Camera Feed
Runs real-time YOLOv8 inference on webcam or IP camera streams.
"""
import cv2
import numpy as np
import time
from collections import deque


def list_available_cameras(max_test: int = 5) -> list[int]:
    """Return list of working camera indices."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


class LiveCamera:
    """
    Real-time detection on webcam or RTSP stream.
    Yields annotated frames and detection dicts for Streamlit display.
    """

    def __init__(self, source=0, conf_threshold=0.3, category_filter=None):
        self.source = source
        self.conf_threshold = conf_threshold
        self.category_filter = category_filter
        self.cap = None
        self.fps_buffer = deque(maxlen=30)
        self._running = False

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def release(self):
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def detect_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """Run YOLO on a single frame. Returns (annotated_frame, detections)."""
        from core.detector import detect_frame
        t0 = time.time()
        detections = detect_frame(frame, self.conf_threshold, self.category_filter)
        elapsed = time.time() - t0
        self.fps_buffer.append(1.0 / max(elapsed, 0.001))

        annotated = _draw_detections(frame.copy(), detections)
        fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        _draw_overlay(annotated, fps, len(detections))
        return annotated, detections

    def get_fps(self) -> float:
        return float(np.mean(self.fps_buffer)) if self.fps_buffer else 0.0

    def snapshot(self) -> tuple[np.ndarray | None, list]:
        """Grab one frame, run detection, return (annotated, detections)."""
        frame = self.read_frame()
        if frame is None:
            return None, []
        return self.detect_frame(frame)


def _draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    COLORS = {
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

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det.get("display_label", det.get("label", "?"))
        conf  = det.get("confidence", 0)
        color = COLORS.get(label.lower(), DEFAULT_COLOR)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def _draw_overlay(frame: np.ndarray, fps: float, det_count: int):
    """Draw FPS and detection count overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Detections: {det_count}", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)


def build_live_detection_record(frame: np.ndarray, detections: list,
                                 frame_idx: int, timestamp_sec: float) -> dict:
    """Package a live frame+detections into the standard frame_data format."""
    m = int(timestamp_sec // 60)
    s = timestamp_sec % 60
    return {
        "frame_idx":    frame_idx,
        "timestamp_sec": round(timestamp_sec, 3),
        "timestamp_str": f"{m:02d}:{s:05.2f}",
        "image":        frame,
        "detections":   detections,
    }
