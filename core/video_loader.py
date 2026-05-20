"""
video_loader.py
Handles video ingestion and frame extraction.
"""
import cv2
import numpy as np
from pathlib import Path


def load_video_metadata(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration   = total_frames / fps if fps > 0 else 0
    cap.release()

    return {
        "path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
        "duration_str": _sec_to_hms(duration),
    }


def extract_frames(video_path: str,
                   sample_fps: float = 2.0,
                   start_sec: float = 0.0,
                   end_sec: float = None) -> list[dict]:
    """
    Extract frames from video at `sample_fps` frames-per-second.
    Returns list of dicts: {frame_idx, timestamp_sec, image (np.ndarray BGR)}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / native_fps if native_fps > 0 else 0

    end_sec = end_sec if end_sec is not None else duration
    end_sec = min(end_sec, duration)

    # Interval in frames between samples
    interval = max(1, int(native_fps / sample_fps))

    frames = []
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts > end_sec:
            break
        if ts < start_sec:
            frame_idx += 1
            continue

        if frame_idx % interval == 0:
            frames.append({
                "frame_idx":    frame_idx,
                "timestamp_sec": round(ts, 3),
                "timestamp_str": _sec_to_hms(ts),
                "image":         frame,
            })

        frame_idx += 1

    cap.release()
    return frames


def _sec_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"
