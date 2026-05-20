"""
batch_processor.py — Multi-Video Batch Search
Ingest multiple videos and run unified prompt search across all of them.
"""

import os
from pathlib import Path


def process_single_video(video_path, sample_fps=2, conf_threshold=0.3,
                          category_filter=None, start_sec=0.0, end_sec=9999.0):
    """
    Run frame extraction + YOLO detection on one video.
    Returns (video_path, detected_frames, inventory).
    """
    from core.video_loader import extract_frames, load_video_metadata
    from core.detector import detect_all_frames
    from core.inventory import build_inventory

    meta = load_video_metadata(video_path)
    end_sec = min(end_sec, meta["duration_sec"])

    frames = extract_frames(
        video_path,
        sample_fps=sample_fps,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    detected = detect_all_frames(
        frames,
        conf_threshold=conf_threshold,
        category_filter=category_filter,
    )

    inventory = build_inventory(detected)

    return {
        "video_path": video_path,
        "video_name": Path(video_path).name,
        "meta": meta,
        "detected_frames": detected,
        "inventory": inventory,
    }


def batch_process_videos(video_paths, sample_fps=2, conf_threshold=0.3,
                          category_filter=None, progress_callback=None):
    """
    Process a list of video paths.
    progress_callback(i, total, video_name) called after each video.
    Returns list of result dicts from process_single_video.
    """
    results = []

    for i, vpath in enumerate(video_paths):
        if progress_callback:
            progress_callback(i, len(video_paths), Path(vpath).name)

        try:
            result = process_single_video(
                vpath,
                sample_fps=sample_fps,
                conf_threshold=conf_threshold,
                category_filter=category_filter,
            )
            results.append(result)
        except Exception as e:
            results.append({
                "video_path": vpath,
                "video_name": Path(vpath).name,
                "error": str(e),
                "detected_frames": [],
                "inventory": None,
            })

    return results


def batch_search(prompt, batch_results, similarity_threshold=0.20):
    """
    Run CLIP prompt search across all processed videos.
    Returns list of {video_name, video_path, matches} dicts.
    """
    from core.clip_matcher import match_prompt_to_frames

    all_results = []

    for res in batch_results:
        if res.get("error") or not res["detected_frames"]:
            all_results.append({
                "video_name": res["video_name"],
                "video_path": res["video_path"],
                "matches": [],
                "error": res.get("error"),
            })
            continue

        matches = match_prompt_to_frames(
            prompt,
            res["detected_frames"],
            similarity_threshold=similarity_threshold,
        )

        all_results.append({
            "video_name": res["video_name"],
            "video_path": res["video_path"],
            "matches": matches,
            "match_count": len(matches),
        })

    # Sort by most matches first
    all_results.sort(key=lambda x: x.get("match_count", 0), reverse=True)

    return all_results


def get_batch_summary(batch_search_results):
    """Summarize batch search results."""
    total_matches = sum(r.get("match_count", 0) for r in batch_search_results)
    videos_with_matches = sum(1 for r in batch_search_results if r.get("match_count", 0) > 0)

    return {
        "total_videos": len(batch_search_results),
        "videos_with_matches": videos_with_matches,
        "total_matches": total_matches,
    }