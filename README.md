# SmartLens AI 🎯 — Finals Edition
### Prompt-Based Video Object Finder | Advanced CV Project

---

## What's New in Finals

| # | Feature | Module | Description |
|---|---------|--------|-------------|
| 1 | **Person Re-Identification** | `core/reid.py` | Assigns persistent IDs to people using CLIP embeddings + cosine similarity. "Track Person A across the video." |
| 2 | **Object Tracking + Trajectories** | `core/tracker.py` | ByteTrack via Ultralytics — draws movement paths for every tracked object. "Show me where this car went." |
| 3 | **Scene/Event Description** | `core/scene_describer.py` | BLIP-2 vision-language model auto-generates natural language descriptions of frames and clips. |
| 4 | **Anomaly Detection** | `core/anomaly_detector.py` | CLIP frame embeddings flagged against statistical distribution — finds unusual events automatically. |
| 5 | **Multi-Video Batch Search** | `core/batch_processor.py` | Upload multiple camera feeds, run unified CLIP prompt search across all simultaneously. |
| 6 | **Export & Reporting** | `core/exporter.py` | One-click PDF or JSON report: inventory, timestamps, clips, anomalies, Re-ID — all downloadable. |
| 7 | **Live Camera / Webcam** | `core/live_camera.py` | Real-time YOLOv8 detection on webcam or RTSP stream, with frame capture for offline analysis. |
| 8 | **Spatial Queries** | `core/spatial_filter.py` | "Person in top-left region" or "car and person within 200px" using bounding box coordinates. |

---

## Full Feature Set

| Feature | Description |
|---------|-------------|
| Prompt-based search | Full sentences or single words via CLIP |
| Object locator | Every timestamp an object appears |
| Clip extraction | Auto-extracts matching video segments |
| Zoom + Enhance | Crops and enhances detected regions |
| Object inventory | All objects in video, flags unidentified |
| Enhancement filters | CLAHE, denoise, deblur, sharpen, brightness, contrast |
| Person Re-ID | CLIP embeddings track individuals across frames |
| ByteTrack | Trajectory paths for every tracked object |
| BLIP-2 scene AI | Natural language descriptions of video content |
| Anomaly detection | Statistical outlier frames flagged automatically |
| Batch search | Multi-video simultaneous search |
| PDF/JSON export | Full downloadable session report |
| Webcam live feed | Real-time YOLO on camera or RTSP stream |
| Spatial queries | Region filter + co-occurrence detection |

---

## Project Structure

```
smartlens/
├── app.py                     ← Streamlit UI — 8-tab interface (run this)
├── core/
│   ├── video_loader.py        ← Frame extraction
│   ├── preprocessor.py        ← Enhancement filters
│   ├── detector.py            ← YOLOv8 + offline HOG fallback
│   ├── clip_matcher.py        ← CLIP prompt matching + TF-IDF fallback
│   ├── clip_extractor.py      ← Video clip extraction + merging
│   ├── inventory.py           ← Full object inventory builder
│   ├── reid.py                ← Person Re-ID (CLIP embeddings + cosine sim)
│   ├── tracker.py             ← ByteTrack object tracking + trajectories
│   ├── scene_describer.py     ← BLIP-2 scene/event description
│   ├── anomaly_detector.py    ← CLIP-based anomaly detection
│   ├── batch_processor.py     ← Multi-video batch search pipeline
│   ├── spatial_filter.py      ← Region + co-occurrence spatial queries
│   ├── live_camera.py         ← Webcam / RTSP real-time detection
│   └── exporter.py            ← PDF + JSON report generation
├── utils/
│   ├── response.py            ← Smart UI status messages
│   └── visualizer.py          ← Annotations, zoom, timeline, trajectories, Re-ID mosaic
├── dataset/
│   └── demo_video.mp4         ← Built-in dataset video
├── outputs/                   ← Extracted clips + reports saved here
└── requirements.txt
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU (optional but recommended for BLIP-2 and faster CLIP):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install faiss-gpu  # instead of faiss-cpu
> ```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## How to Use — Tab by Tab

| Tab | What to do |
|-----|-----------|
| **🔍 Scan & Search** | Load video → Scan → Search with prompts → Extract clips |
| **🧑 Re-ID & Tracking** | After scan: Run Re-ID to identify unique persons; Run Tracking to see trajectories |
| **🚨 Anomaly Detection** | After scan: Detect statistically unusual frames automatically |
| **🗣 Scene Description** | After scan or clip extraction: Generate natural language descriptions using BLIP-2 |
| **📁 Batch Search** | Upload 2+ videos → Enter prompt → Search all simultaneously |
| **📡 Live Camera** | Connect webcam → Capture frames → Run real-time detection |
| **🗺 Spatial Queries** | After scan: Filter by screen region or find co-occurring objects |
| **📄 Export & Report** | After any analysis: Download PDF or JSON report of results |

---

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| **YOLOv8n** | Object detection + ByteTrack tracking | ~6MB |
| **CLIP ViT-B/32** | Prompt matching + Re-ID embeddings + Anomaly detection | ~340MB |
| **BLIP-2 (OPT-2.7B)** | Scene/event description | ~2GB (downloads on first use) |
| **HOG** | Offline fallback detection (no download needed) | Built into OpenCV |

---

## Fallback Modes (No Internet Required)

- **Detection:** YOLOv8 → falls back to HOG + background subtraction
- **Prompt matching:** CLIP → falls back to TF-IDF + keyword synonyms
- **All core features work offline** except BLIP-2 (needs initial download)

---

## Group Members & Responsibilities

| Member | Responsibilities |
|--------|-----------------|
| Member 1 | `video_loader.py`, `preprocessor.py`, `live_camera.py`, dataset, augmentation |
| Member 2 | `detector.py`, `clip_matcher.py`, `inventory.py`, `anomaly_detector.py`, `reid.py` |
| Member 3 | `clip_extractor.py`, `tracker.py`, `scene_describer.py`, `batch_processor.py`, `spatial_filter.py`, `exporter.py`, `app.py` UI, `visualizer.py`, `response.py` |

---

## Tech Stack

`Python 3.10+` · `YOLOv8` · `ByteTrack` · `CLIP` · `BLIP-2` · `FAISS` · `OpenCV` · `Streamlit` · `PyTorch` · `Transformers` · `NumPy` · `Pillow` · `fpdf2` · `scikit-learn`
