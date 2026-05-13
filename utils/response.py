"""
response.py
Smart response messages for all match outcomes.
"""


def get_response(query: str,
                 matches: list,
                 mode: str = "prompt",
                 inventory: dict = None) -> dict:
    """
    Generate a human-readable status message based on match results.

    mode: "prompt" | "locate" | "inventory"
    Returns: { status: "success"|"partial"|"not_found"|"error", message: str, suggestion: str }
    """
    count = len(matches)

    # ── INVENTORY MODE ──
    if mode == "inventory":
        if not inventory or inventory["summary"]["unique_objects"] == 0:
            return {
                "status":     "not_found",
                "message":    "No objects detected in this video.",
                "suggestion": "Try adjusting the confidence threshold or check the video quality.",
            }
        n = inventory["summary"]["unique_objects"]
        u = inventory["summary"]["unidentified_count"]
        msg = f"Found {n} unique object type{'s' if n != 1 else ''} across the video."
        if u > 0:
            msg += f" {u} detection{'s' if u != 1 else ''} could not be confidently identified."
        return {"status": "success", "message": msg, "suggestion": ""}

    # ── NOT FOUND ──
    if count == 0:
        known_objects = []
        if inventory and inventory.get("objects"):
            known_objects = list(inventory["objects"].keys())

        suggestion = ""
        if known_objects:
            suggestion = (
                f"Objects detected in this video include: "
                f"{', '.join(known_objects[:6])}. "
                f"Try searching for one of these."
            )
        else:
            suggestion = (
                "No objects were detected matching your query. "
                "Try lowering the confidence threshold or broadening your search."
            )

        return {
            "status":     "not_found",
            "message":    f"No matches found for \"{query}\" in this video.",
            "suggestion": suggestion,
        }

    # ── PARTIAL (1-2 matches, low similarity) ──
    if count <= 2 and mode == "prompt":
        avg_sim = sum(m.get("clip_similarity", 0) for m in matches) / count
        if avg_sim < 0.22:
            return {
                "status":  "partial",
                "message": f"Found {count} weak match{'es' if count != 1 else ''} for \"{query}\" "
                           f"(low confidence). Results may not be accurate.",
                "suggestion": "Try rephrasing your prompt or lowering the similarity threshold.",
            }

    # ── SUCCESS ──
    timestamps = sorted(set(m["timestamp_str"] for m in matches))
    ts_preview = ", ".join(timestamps[:4])
    if len(timestamps) > 4:
        ts_preview += f" ... (+{len(timestamps)-4} more)"

    label_counts = {}
    for m in matches:
        l = m.get("label", "object")
        label_counts[l] = label_counts.get(l, 0) + 1

    label_str = ", ".join(
        f"{v}× {k}" for k, v in sorted(label_counts.items(), key=lambda x: -x[1])
    )

    return {
        "status":  "success",
        "message": f"Found {count} match{'es' if count != 1 else ''} for \"{query}\" "
                   f"({label_str}) at: {ts_preview}",
        "suggestion": "",
    }


def get_clip_response(clips: list, query: str) -> dict:
    """Response after clip extraction."""
    if not clips:
        return {
            "status":     "not_found",
            "message":    "No clips could be extracted.",
            "suggestion": "Matches were found but clip extraction failed. Check output directory permissions.",
        }

    total_duration = sum((c["end_sec"] - c["start_sec"]) for c in clips)
    return {
        "status":  "success",
        "message": f"Extracted {len(clips)} clip{'s' if len(clips)!=1 else ''} for \"{query}\" "
                   f"(~{total_duration:.1f}s total footage).",
        "suggestion": "",
    }
