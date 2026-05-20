"""
scene_describer.py — Scene/Event Description using BLIP-2
Generates natural language descriptions of video clips or key frames.
"""

import cv2
import numpy as np
from PIL import Image


_blip_model = None
_blip_processor = None
_blip_device = None


def load_blip():
    """Load BLIP-2 model (downloads on first run ~2GB)."""
    global _blip_model, _blip_processor, _blip_device
    if _blip_model is None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch

        _blip_device = "cuda" if torch.cuda.is_available() else "cpu"

        _blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        _blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if _blip_device == "cuda" else torch.float32,
        ).to(_blip_device)

    return _blip_model, _blip_processor, _blip_device


def describe_frame(frame_bgr, question=None):
    """
    Generate a natural language description of a single frame.
    Optionally ask a specific question about the frame.
    """
    import torch

    model, processor, device = load_blip()

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    if question:
        prompt = f"Question: {question} Answer:"
        inputs = processor(img, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(img, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    description = processor.decode(output[0], skip_special_tokens=True).strip()
    return description


def describe_clip(clip_path, num_keyframes=5, question=None):
    """
    Sample keyframes from a clip and generate a combined description.
    Returns per-frame descriptions and an aggregated summary.
    """
    cap = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return {"error": "Could not read clip.", "descriptions": [], "summary": ""}

    indices = np.linspace(0, total - 1, num=min(num_keyframes, total), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    descriptions = []
    for i, frame in enumerate(frames):
        desc = describe_frame(frame, question=question)
        descriptions.append({
            "keyframe_index": i,
            "description": desc,
        })

    # Combine into a single summary (just join unique sentences)
    all_text = " ".join([d["description"] for d in descriptions])
    summary = all_text  # Can be further summarized with an LLM if needed

    return {
        "descriptions": descriptions,
        "summary": summary,
        "keyframes_used": len(frames),
    }


def describe_detections(detected_frames, max_frames=8):
    """
    Pick the highest-confidence frames from detected_frames and describe them.
    Returns list of {timestamp_str, description}.
    """
    sorted_frames = sorted(detected_frames, key=lambda x: x.get("confidence", 0), reverse=True)
    top_frames = sorted_frames[:max_frames]

    results = []
    for det in top_frames:
        desc = describe_frame(det["image"])
        results.append({
            "timestamp_str": det.get("timestamp_str", "??"),
            "label": det.get("label", "unknown"),
            "description": desc,
        })

    return results