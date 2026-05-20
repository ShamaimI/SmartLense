"""
reid.py — Person Re-Identification
"""
import cv2
import numpy as np
from collections import defaultdict


def extract_crop(frame, bbox, padding=10):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


_embed_model = None
_embed_preprocess = None
_embed_device = None

def get_embed_model():
    global _embed_model, _embed_preprocess, _embed_device
    if _embed_model is None:
        import clip, torch
        _embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model, _embed_preprocess = clip.load("ViT-B/32", device=_embed_device)
    return _embed_model, _embed_preprocess, _embed_device


def embed_crop(crop_bgr):
    import torch
    from PIL import Image

    model, preprocess, device = get_embed_model()
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class PersonReID:
    def __init__(self, similarity_threshold=0.82):
        self.similarity_threshold = similarity_threshold
        self.gallery = {}
        self.next_id = 1
        self.person_tracks = defaultdict(list)

    def reset(self):
        self.gallery.clear()
        self.person_tracks.clear()
        self.next_id = 1

    def assign_id(self, embedding):
        best_id, best_sim = None, -1
        for pid, gal_emb in self.gallery.items():
            sim = cosine_similarity(embedding, gal_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        if best_sim >= self.similarity_threshold:
            self.gallery[best_id] = (self.gallery[best_id] + embedding) / 2
            return best_id, best_sim
        else:
            new_id = self.next_id
            self.gallery[new_id] = embedding
            self.next_id += 1
            return new_id, best_sim

    def process_detections(self, flat_detections):
        """
        Expects flat list of detection dicts (each with 'image', 'bbox', 'label').
        Use flatten_detections() in app.py before calling this.
        """
        self.reset()
        results = []

        for det in flat_detections:
            if det["label"] != "person":
                det["person_id"] = None
                results.append(det)
                continue

            crop = extract_crop(det["image"], det["bbox"])
            embedding = embed_crop(crop)

            if embedding is None:
                det["person_id"] = None
                results.append(det)
                continue

            pid, sim = self.assign_id(embedding)
            det["person_id"] = pid
            det["reid_similarity"] = sim
            self.person_tracks[pid].append(det)
            results.append(det)

        return results

    def get_person_summary(self):
        summary = {}
        for pid, detections in self.person_tracks.items():
            timestamps = [d["timestamp_str"] for d in detections]
            summary[pid] = {
                "count":      len(detections),
                "first_seen": timestamps[0] if timestamps else "N/A",
                "last_seen":  timestamps[-1] if timestamps else "N/A",
                "timestamps": timestamps,
            }
        return summary


_reid_instance = None

def get_reid():
    global _reid_instance
    if _reid_instance is None:
        _reid_instance = PersonReID()
    return _reid_instance