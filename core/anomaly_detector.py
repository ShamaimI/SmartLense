"""
anomaly_detector.py — Anomaly Detection using CLIP embeddings
Operates on frame-level dicts (output of extract_frames + detect_all_frames).
"""
import numpy as np


def get_frame_embeddings(frame_images):
    """
    Encode a list of raw BGR numpy frames using CLIP.
    Returns numpy array shape (N, embedding_dim).
    """
    import torch, clip, cv2
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    embeddings = []
    for frame_bgr in frame_images:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy()[0])

    return np.array(embeddings)


def compute_anomaly_scores(embeddings):
    mean_emb = embeddings.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    scores = []
    for emb in embeddings:
        sim = float(np.dot(emb, mean_emb))
        scores.append(1.0 - sim)
    return np.array(scores)


def detect_anomalies(detected_frames, threshold_percentile=90):
    """
    detected_frames: list of frame dicts from detect_all_frames()
    Each dict has: image, timestamp_sec, timestamp_str, detections, frame_idx
    Returns list of anomalous frame dicts with anomaly_score added.
    """
    if not detected_frames:
        return []

    raw_frames = [f["image"] for f in detected_frames]
    embeddings = get_frame_embeddings(raw_frames)
    scores = compute_anomaly_scores(embeddings)

    threshold = np.percentile(scores, threshold_percentile)

    anomalies = []
    for frame_data, score in zip(detected_frames, scores):
        if score >= threshold:
            entry = {**frame_data, "anomaly_score": float(score)}
            anomalies.append(entry)

    anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
    return anomalies


def get_anomaly_summary(anomalies):
    if not anomalies:
        return {"count": 0, "top_timestamps": []}
    return {
        "count":           len(anomalies),
        "top_timestamps":  [a["timestamp_str"] for a in anomalies[:5]],
        "max_score":       round(anomalies[0]["anomaly_score"], 4),
        "min_score":       round(anomalies[-1]["anomaly_score"], 4),
    }