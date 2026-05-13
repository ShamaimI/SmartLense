"""
clip_extractor.py
Extracts video clips from matched timestamps and saves them.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from core.preprocessor import apply_filters


def extract_clip(video_path: str,
                 start_sec: float,
                 end_sec: float,
                 output_path: str,
                 filters: dict = None,
                 padding_sec: float = 1.0) -> str | None:
    """
    Extract a clip from video between start_sec and end_sec.
    Applies enhancement filters if provided.
    Returns output path if successful, None otherwise.
    """
    filters = filters or {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0

    # Add padding around the clip
    clip_start = max(0.0, start_sec - padding_sec)
    clip_end   = min(duration, end_sec + padding_sec)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts > clip_end:
            break

        # Apply enhancement filters
        if any(v for v in filters.values()):
            frame = apply_filters(frame, filters)

        out.write(frame)
        written += 1

    cap.release()
    out.release()

    return output_path if written > 0 else None


def extract_clips_from_matches(video_path: str,
                                matches: list[dict],
                                output_dir: str,
                                filters: dict = None,
                                padding_sec: float = 1.0,
                                merge_gap_sec: float = 2.0) -> list[dict]:
    """
    Extract clips for all matches, merging nearby timestamps into single clips.
    Returns list of clip dicts with paths and metadata.
    """
    if not matches:
        return []

    # Group nearby timestamps into clip segments
    segments = _merge_timestamps(matches, merge_gap_sec)

    clips = []
    for i, seg in enumerate(segments):
        filename = f"clip_{i+1:03d}_{seg['start_sec']:.1f}s.mp4"
        out_path = str(Path(output_dir) / filename)

        result_path = extract_clip(
            video_path=video_path,
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            output_path=out_path,
            filters=filters,
            padding_sec=padding_sec,
        )

        if result_path:
            clips.append({
                "clip_path":    result_path,
                "start_sec":    seg["start_sec"],
                "end_sec":      seg["end_sec"],
                "start_str":    _sec_to_str(seg["start_sec"]),
                "end_str":      _sec_to_str(seg["end_sec"]),
                "match_count":  seg["count"],
                "best_match":   seg["best_match"],
                "labels":       seg["labels"],
                "avg_similarity": seg["avg_sim"],
                "thumbnail":    seg["thumbnail"],
            })

    return clips


def _merge_timestamps(matches: list[dict], gap_sec: float) -> list[dict]:
    """Group nearby timestamp hits into merged segments."""
    if not matches:
        return []

    sorted_m = sorted(matches, key=lambda x: x["timestamp_sec"])
    segments = []
    seg = {
        "start_sec": sorted_m[0]["timestamp_sec"],
        "end_sec":   sorted_m[0]["timestamp_sec"],
        "count":     1,
        "best_match": sorted_m[0],
        "labels":    {sorted_m[0]["label"]},
        "sims":      [sorted_m[0]["clip_similarity"]],
        "thumbnail": sorted_m[0]["image"],
    }

    for m in sorted_m[1:]:
        if m["timestamp_sec"] - seg["end_sec"] <= gap_sec:
            seg["end_sec"] = m["timestamp_sec"]
            seg["count"]  += 1
            seg["labels"].add(m["label"])
            seg["sims"].append(m["clip_similarity"])
            if m["clip_similarity"] > seg["best_match"]["clip_similarity"]:
                seg["best_match"] = m
                seg["thumbnail"]  = m["image"]
        else:
            seg["avg_sim"] = round(sum(seg["sims"]) / len(seg["sims"]), 3)
            segments.append(seg)
            seg = {
                "start_sec": m["timestamp_sec"],
                "end_sec":   m["timestamp_sec"],
                "count":     1,
                "best_match": m,
                "labels":    {m["label"]},
                "sims":      [m["clip_similarity"]],
                "thumbnail": m["image"],
            }

    seg["avg_sim"] = round(sum(seg["sims"]) / len(seg["sims"]), 3)
    segments.append(seg)
    return segments


def _sec_to_str(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"
