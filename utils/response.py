"""
response.py — Smart status/error messages for the UI.
"""


def get_response(prompt: str, matches: list, mode: str = "prompt",
                 inventory: dict = None) -> dict:
    """
    Return a status dict: { status, message, suggestion }
    status: "success" | "partial" | "not_found" | "error"
    """
    if not matches:
        suggestion = ""
        if inventory:
            objects = list(inventory.get("objects", {}).keys())
            if objects:
                suggestion = f"Tip: Objects found in this video include: {', '.join(objects[:5])}."
        return {
            "status": "not_found",
            "message": f"No matches found for \"{prompt}\".",
            "suggestion": suggestion or "Try adjusting the confidence or similarity threshold.",
        }

    count = len(matches)
    avg_sim = sum(m.get("clip_similarity", 0) for m in matches) / count

    if mode == "locate":
        timestamps = sorted(set(m["timestamp_str"] for m in matches))
        if count >= 5:
            return {
                "status": "success",
                "message": f"Found \"{prompt}\" at {len(timestamps)} timestamp(s) across the video.",
                "suggestion": "",
            }
        return {
            "status": "partial",
            "message": f"Found \"{prompt}\" at {len(timestamps)} location(s). Coverage may be partial.",
            "suggestion": "Try lowering the confidence threshold for more results.",
        }

    if avg_sim >= 0.35 and count >= 3:
        return {
            "status": "success",
            "message": f"Strong match! Found {count} frame(s) for \"{prompt}\" (avg similarity {avg_sim:.2f}).",
            "suggestion": "",
        }
    elif count >= 1:
        return {
            "status": "partial",
            "message": f"Found {count} partial match(es) for \"{prompt}\" (avg similarity {avg_sim:.2f}).",
            "suggestion": "Consider lowering the CLIP similarity threshold for broader results.",
        }
    return {
        "status": "not_found",
        "message": f"No strong matches for \"{prompt}\".",
        "suggestion": "Try a simpler prompt or lower the threshold.",
    }


def get_clip_response(clips: list, prompt: str) -> dict:
    if not clips:
        return {
            "status": "error",
            "message": "No clips could be extracted.",
            "suggestion": "Check that the video file is accessible and try again.",
        }
    total_sec = sum(c.get("end_sec", 0) - c.get("start_sec", 0) for c in clips)
    return {
        "status": "success",
        "message": f"Extracted {len(clips)} clip(s) totalling ~{total_sec:.1f}s for \"{prompt}\".",
        "suggestion": "",
    }
