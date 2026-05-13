"""
clip_matcher.py
Dual-mode text-to-video prompt matching:
  - PRIMARY:  OpenAI CLIP ViT-B/32 (when model weights are downloadable)
  - FALLBACK: TF-IDF cosine similarity + keyword expansion (100% offline)

Supports both single words ("car") and full natural-language sentences
("person carrying a bag near the door").
"""
import numpy as np
import cv2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
#  CLIP PRIMARY  (optional – used when weights can be downloaded)
# ─────────────────────────────────────────────────────────────────────────────
_clip_model      = None
_clip_preprocess = None
_clip_available  = None
_device          = "cpu"


def _try_load_clip():
    global _clip_model, _clip_preprocess, _clip_available, _device
    if _clip_available is not None:
        return _clip_available
    try:
        import torch
        import clip
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)
        _clip_available = True
        print("[clip_matcher] CLIP loaded.")
    except Exception as e:
        _clip_available = False
        print(f"[clip_matcher] CLIP unavailable ({e}). Using TF-IDF offline matcher.")
    return _clip_available


def _clip_similarity(prompt: str, frame: np.ndarray, bbox: tuple) -> float:
    import torch, clip
    from PIL import Image
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0:
        crop = frame
    pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_t  = _clip_preprocess(pil).unsqueeze(0).to(_device)
    tokens = clip.tokenize([prompt]).to(_device)
    with torch.no_grad():
        img_emb  = _clip_model.encode_image(img_t)
        txt_emb  = _clip_model.encode_text(tokens)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
    return float((txt_emb @ img_emb.T).item())


# ─────────────────────────────────────────────────────────────────────────────
#  OFFLINE FALLBACK: TF-IDF + KEYWORD EXPANSION
# ─────────────────────────────────────────────────────────────────────────────

# Synonym / alias map – expands user prompts so "human" matches "person" etc.
SYNONYMS = {
    "human":      ["person", "people", "man", "woman", "pedestrian"],
    "people":     ["person", "man", "woman", "pedestrian", "human"],
    "man":        ["person", "male", "human"],
    "woman":      ["person", "female", "human"],
    "kid":        ["person", "child"],
    "child":      ["person", "kid"],
    "vehicle":    ["car", "truck", "bus", "motorbike", "bicycle"],
    "automobile": ["car", "vehicle"],
    "auto":       ["car", "vehicle"],
    "bike":       ["bicycle", "motorbike"],
    "motorbike":  ["motorcycle", "bike"],
    "purse":      ["bag", "handbag", "backpack"],
    "luggage":    ["bag", "suitcase", "backpack"],
    "walking":    ["person"],
    "running":    ["person"],
    "driving":    ["car", "vehicle"],
    "moving":     ["car", "person", "vehicle"],
    "standing":   ["person"],
    "sitting":    ["person"],
    "carrying":   ["person", "bag"],
}

# Context descriptors that enrich the label corpus
LABEL_DESCRIPTORS = {
    "person":    "person human walking standing running carrying pedestrian man woman",
    "car":       "car vehicle automobile driving moving road sedan hatchback",
    "truck":     "truck vehicle large heavy transport cargo moving road",
    "bus":       "bus vehicle large public transport passengers road",
    "bicycle":   "bicycle bike cycling two wheels rider person",
    "motorbike": "motorbike motorcycle bike rider two wheels fast",
    "bag":       "bag backpack purse luggage carrying handbag object",
    "backpack":  "backpack bag carrying person school luggage",
    "object":    "object unidentified unknown item thing",
}


def _expand_prompt(prompt: str) -> str:
    """Expand prompt using synonym map for better matching."""
    words  = re.findall(r'\w+', prompt.lower())
    extra  = []
    for w in words:
        if w in SYNONYMS:
            extra.extend(SYNONYMS[w])
    return prompt + " " + " ".join(extra)


def _tfidf_similarity(prompt: str, label: str) -> float:
    """TF-IDF cosine similarity between expanded prompt and label descriptor."""
    descriptor = LABEL_DESCRIPTORS.get(label.lower(), label)
    expanded   = _expand_prompt(prompt)
    try:
        vec  = TfidfVectorizer(ngram_range=(1, 2))
        X    = vec.fit_transform([expanded, descriptor])
        sim  = cosine_similarity(X[0:1], X[1:2])[0][0]
        return float(sim)
    except Exception:
        # word-overlap fallback
        p_words = set(re.findall(r'\w+', expanded.lower()))
        l_words = set(re.findall(r'\w+', descriptor.lower()))
        if not p_words or not l_words:
            return 0.0
        return len(p_words & l_words) / len(p_words | l_words)


def _label_in_prompt(prompt: str, label: str) -> float:
    """Direct keyword check — highest priority signal."""
    p = prompt.lower()
    l = label.lower()
    if l in p:
        return 0.9
    # Check synonyms
    syns = SYNONYMS.get(l, [])
    for word in re.findall(r'\w+', p):
        if word == l or word in syns:
            return 0.82
        if l in SYNONYMS.get(word, []):
            return 0.75
    return 0.0


def _offline_similarity(prompt: str, label: str) -> float:
    """Combined offline similarity score."""
    direct = _label_in_prompt(prompt, label)
    if direct > 0:
        return direct
    tfidf = _tfidf_similarity(prompt, label)
    return tfidf


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def match_prompt_to_frames(prompt: str,
                            detected_frames: list[dict],
                            similarity_threshold: float = 0.20) -> list[dict]:
    """
    Match a text prompt against all detections in all frames.
    Returns matched detections sorted by similarity (best first).
    """
    use_clip = _try_load_clip()
    matches  = []

    for frame_data in detected_frames:
        frame = frame_data["image"]
        for det in frame_data.get("detections", []):
            if use_clip:
                sim = _clip_similarity(prompt, frame, det["bbox"])
            else:
                sim = _offline_similarity(prompt, det["label"])

            if sim >= similarity_threshold:
                matches.append({
                    "timestamp_sec":   frame_data["timestamp_sec"],
                    "timestamp_str":   frame_data["timestamp_str"],
                    "frame_idx":       frame_data["frame_idx"],
                    "label":           det["display_label"],
                    "confidence":      det["confidence"],
                    "bbox":            det["bbox"],
                    "clip_similarity": round(sim, 4),
                    "image":           frame,
                })

    matches.sort(key=lambda x: x["clip_similarity"], reverse=True)

    # Deduplicate: if same timestamp + bbox already in list, skip
    seen, deduped = set(), []
    for m in matches:
        key = (m["timestamp_sec"], m["bbox"])
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    return deduped


def locate_object(object_name: str,
                  detected_frames: list[dict],
                  conf_threshold: float = 0.3) -> list[dict]:
    """
    Find every timestamp where a named object appears.
    Uses direct label matching first, falls back to semantic similarity.
    """
    hits       = []
    name_lower = object_name.lower().strip()

    for frame_data in detected_frames:
        for det in frame_data.get("detections", []):
            label_lower = det["label"].lower()
            # Direct match
            direct = name_lower in label_lower or label_lower in name_lower
            # Synonym match
            syn_match = any(
                name_lower in syns or label_lower in SYNONYMS.get(name_lower, [])
                for syns in [SYNONYMS.get(name_lower, [])]
            )
            if (direct or syn_match) and det["confidence"] >= conf_threshold:
                hits.append({
                    "timestamp_sec":   frame_data["timestamp_sec"],
                    "timestamp_str":   frame_data["timestamp_str"],
                    "frame_idx":       frame_data["frame_idx"],
                    "label":           det["display_label"],
                    "confidence":      det["confidence"],
                    "bbox":            det["bbox"],
                    "clip_similarity": 1.0,
                    "image":           frame_data["image"],
                })

    # If no direct hits, fall back to semantic matching
    if not hits:
        hits = match_prompt_to_frames(object_name, detected_frames,
                                       similarity_threshold=0.18)

    hits.sort(key=lambda x: x["timestamp_sec"])
    return hits
