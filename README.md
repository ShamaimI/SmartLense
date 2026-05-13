# SmartLens AI 🎯
### Prompt-Based Video Object Finder | CV Project

---

## What It Does

SmartLens lets you search video footage using natural language prompts.  
Type *"person carrying a bag"* or just *"car"* — it scans the video, finds every matching moment, extracts those clips, and lets you zoom into the detected region with enhancement filters.

**Use cases:** CCTV surveillance · Baby monitors · Traffic monitoring

---

## Features

| Feature | Description |
|---|---|
| **Prompt-based search** | Full sentences or single words via CLIP |
| **Object locator** | Find every timestamp an object appears |
| **Clip extraction** | Auto-extracts matching video segments |
| **Zoom + Enhance** | Crops and enhances detected regions |
| **Object inventory** | Lists all objects in video, flags unidentified |
| **Smart responses** | Meaningful messages when nothing is found |
| **Enhancement filters** | CLAHE, denoise, deblur, sharpen, brightness, contrast |
| **Detection filters** | Confidence threshold, category, time range |

---

## Project Structure

```
smartlens/
├── app.py                   ← Streamlit UI (run this)
├── core/
│   ├── video_loader.py      ← Frame extraction
│   ├── preprocessor.py      ← Enhancement filters
│   ├── detector.py          ← YOLOv8 detection
│   ├── clip_matcher.py      ← CLIP prompt matching
│   ├── clip_extractor.py    ← Video clip extraction
│   └── inventory.py         ← Full object inventory
├── utils/
│   ├── response.py          ← Smart error/status messages
│   └── visualizer.py        ← Bounding boxes, zoom, timeline
├── dataset/
│   └── demo_video.mp4       ← Built-in dataset video
├── outputs/                 ← Extracted clips saved here
└── requirements.txt
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** CLIP requires PyTorch. For GPU support install the CUDA version of torch first:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
Streamlit will open automatically at `http://localhost:8501`

---

## How to Use

1. **Load video** — Use the built-in dataset video or upload your own (mp4/avi/mov)
2. **Set filters** — Adjust confidence, FPS, category, time range in the sidebar
3. **Scan video** — Click "Scan Video for Objects" to run YOLOv8 detection
4. **View inventory** — See all detected objects with timestamps
5. **Enter a prompt** — Type anything: `"person"`, `"car moving fast"`, `"bicycle near the road"`
6. **Search or Locate** — 
   - **Search & Extract Clips** → CLIP matches prompt to frames, extracts video clips
   - **Locate Object Across Video** → Finds every timestamp the object appears
7. **View results** — Watch extracted clips, see zoom panels, check the timeline

---

## Models Used

| Model | Purpose |
|---|---|
| **YOLOv8n** (Ultralytics) | Real-time object detection on frames |
| **CLIP ViT-B/32** (OpenAI) | Text-to-image prompt matching |
| **OpenCV** | Frame extraction, enhancement, clip writing |

---

## Dataset

The built-in `dataset/demo_video.mp4` is a synthetic video containing:
- People, cars, trucks, bicycles, bags moving across the scene
- 15 seconds · 20fps · 640×480

To use your own video: uncheck "Use built-in dataset video" and upload any mp4/avi file.

---

## Group Members & Responsibilities

| Member | Responsibilities |
|---|---|
| Member 1 | `video_loader.py`, `preprocessor.py`, dataset, augmentation |
| Member 2 | `detector.py`, `clip_matcher.py`, `inventory.py`, model training |
| Member 3 | `clip_extractor.py`, `visualizer.py`, `app.py` UI, `response.py` |

---

## Tech Stack

`Python 3.10+` · `YOLOv8` · `CLIP` · `OpenCV` · `Streamlit` · `PyTorch` · `NumPy` · `Pillow`
