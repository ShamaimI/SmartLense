"""
tracker.py — Object Tracking with ByteTrack (via Ultralytics)
"""
import numpy as np
from collections import defaultdict


def _sec_to_str(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


class ObjectTracker:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.trajectories = defaultdict(list)

    def track_video(self, video_path, sample_fps=2, conf_threshold=0.3,
                    category_filter=None, start_sec=0.0, end_sec=9999.0):
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(1, int(fps / sample_fps))

        results_all = []
        frame_idx = 0
        self.trajectories.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / fps

            if timestamp_sec < start_sec:
                frame_idx += 1
                continue
            if timestamp_sec > end_sec:
                break

            if frame_idx % frame_interval == 0:
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=conf_threshold,
                    verbose=False,
                )

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]

                        if category_filter and label not in category_filter:
                            continue

                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        track_id = int(box.id[0]) if box.id is not None else -1

                        x_center = int((xyxy[0] + xyxy[2]) / 2)
                        y_center = int((xyxy[1] + xyxy[3]) / 2)

                        self.trajectories[track_id].append((x_center, y_center, timestamp_sec))

                        results_all.append({
                            "track_id":      track_id,
                            "label":         label,
                            "display_label": label,
                            "confidence":    conf,
                            "bbox":          xyxy.tolist(),
                            "identified":    conf >= 0.35,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "timestamp_str": _sec_to_str(timestamp_sec),
                            "image":         frame.copy(),
                            "person_id":     None,
                            "clip_similarity": 1.0,
                        })

            frame_idx += 1

        cap.release()
        return results_all

    def get_trajectories(self):
        return dict(self.trajectories)

    def get_unique_tracks(self, tracked_detections):
        summary = defaultdict(lambda: {"label": "", "count": 0, "timestamps": []})
        for d in tracked_detections:
            tid = d["track_id"]
            summary[tid]["label"] = d["label"]
            summary[tid]["count"] += 1
            summary[tid]["timestamps"].append(d["timestamp_str"])
        return dict(summary)


_tracker_instance = None

def get_tracker():
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ObjectTracker()
    return _tracker_instance