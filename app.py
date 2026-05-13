"""
app.py — SmartLens AI
Streamlit UI for prompt-based video object detection, clip extraction, and enhancement.

Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartLens AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&family=DM+Sans:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #07090d; color: #cce8f4; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0c0f14 !important;
    border-right: 1px solid #1a2535 !important;
}

/* Header */
.sl-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #00e5ff;
    letter-spacing: 3px;
    text-shadow: 0 0 30px rgba(0,229,255,0.4);
    margin-bottom: 0;
}
.sl-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #3a5568;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
    margin-bottom: 20px;
}

/* Status badges */
.badge-success { background:#0d2a1a; border:1px solid #00ff88; color:#00ff88;
    padding:6px 14px; border-radius:3px; font-family:'Share Tech Mono',monospace;
    font-size:12px; display:inline-block; }
.badge-warn    { background:#2a1e00; border:1px solid #ffaa00; color:#ffaa00;
    padding:6px 14px; border-radius:3px; font-family:'Share Tech Mono',monospace;
    font-size:12px; display:inline-block; }
.badge-error   { background:#2a0d14; border:1px solid #ff3355; color:#ff3355;
    padding:6px 14px; border-radius:3px; font-family:'Share Tech Mono',monospace;
    font-size:12px; display:inline-block; }

/* Clip card */
.clip-card {
    background: #0d1219; border: 1px solid #1a2535;
    border-radius: 4px; padding: 12px; margin-bottom: 10px;
}
.clip-ts {
    font-family: 'Share Tech Mono', monospace;
    color: #00e5ff; font-size: 13px;
}
.clip-label { color: #cce8f4; font-size: 14px; font-weight: 600; }
.clip-conf  { color: #3a5568; font-size: 12px; font-family: 'Share Tech Mono', monospace; }

/* Section headers */
.section-head {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #00e5ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2535;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* Inventory table */
.inv-row { display:flex; justify-content:space-between; align-items:center;
    padding: 8px 0; border-bottom:1px solid #111820; font-size:13px; }
.inv-label { font-weight:600; color:#cce8f4; }
.inv-count { font-family:'Share Tech Mono',monospace; color:#00e5ff; }
.inv-conf  { font-family:'Share Tech Mono',monospace; color:#3a5568; font-size:11px; }
.inv-time  { font-family:'Share Tech Mono',monospace; color:#ffaa00; font-size:11px; }
.unid-tag  { background:#1a0d20; border:1px solid #aa66ff; color:#aa66ff;
    font-size:10px; padding:2px 8px; border-radius:2px;
    font-family:'Share Tech Mono',monospace; }

/* Timeline */
.tl-wrap { background:#0a0d12; border:1px solid #1a2535; border-radius:3px;
    padding:6px 10px; margin:10px 0; }
</style>
""", unsafe_allow_html=True)


# ── Imports with spinner ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models (first run may take ~30s)...")
def load_models():
    from core.detector import get_model
    from core.clip_matcher import get_clip
    get_model()
    get_clip()
    return True


# ── Helper: session state defaults ───────────────────────────────────────────
def init_state():
    defaults = {
        "video_path": None,
        "video_meta": None,
        "detected_frames": None,
        "inventory": None,
        "last_query": "",
        "last_matches": [],
        "last_clips": [],
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sl-header" style="font-size:1.5rem">SMART<span style="color:#fff">LENS</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sl-sub">CV PROJECT · AI VIDEO ANALYSIS</div>', unsafe_allow_html=True)
    st.divider()

    # ── Video Upload ──
    st.markdown('<div class="section-head">📁 VIDEO INPUT</div>', unsafe_allow_html=True)

    # Default dataset video
    DATASET_VIDEO = str(Path(__file__).parent / "dataset" / "demo_video.mp4")
    use_dataset = st.checkbox("Use built-in dataset video", value=True)

    if use_dataset:
        if Path(DATASET_VIDEO).exists():
            st.success("Dataset video loaded ✓")
            video_source = DATASET_VIDEO
        else:
            st.error("Dataset video not found.")
            video_source = None
    else:
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.flush()
            video_source = tmp.name
        else:
            video_source = None

    st.divider()

    # ── Detection Filters ──
    st.markdown('<div class="section-head">🔧 DETECTION FILTERS</div>', unsafe_allow_html=True)

    sample_fps = st.slider("Frame sampling (fps)", 1, 10, 2,
                           help="How many frames per second to analyse. Higher = slower but more accurate.")
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05,
                               help="Minimum YOLO detection confidence to keep.")
    similarity_threshold = st.slider("CLIP similarity threshold", 0.10, 0.50, 0.20, 0.01,
                                     help="How strictly the prompt must match. Lower = more results.")

    CATEGORIES = ["All", "person", "car", "truck", "bus", "bicycle",
                  "motorbike", "bag", "backpack"]
    category = st.selectbox("Object category filter", CATEGORIES)
    cat_filter = None if category == "All" else [category]

    st.divider()

    # ── Enhancement Filters ──
    st.markdown('<div class="section-head">✨ ENHANCEMENT FILTERS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        use_clahe   = st.checkbox("Low-light fix (CLAHE)", value=False)
        use_denoise = st.checkbox("Denoise", value=False)
    with col2:
        use_deblur  = st.checkbox("Deblur", value=False)
        use_sharpen = st.checkbox("Sharpen", value=False)

    brightness = st.slider("Brightness", -80, 80, 0)
    contrast   = st.slider("Contrast",   -80, 80, 0)

    enhancement_filters = {
        "clahe":      use_clahe,
        "denoise":    use_denoise,
        "deblur":     use_deblur,
        "sharpen":    use_sharpen,
        "brightness": brightness,
        "contrast":   contrast,
    }

    st.divider()

    # ── Time Range ──
    st.markdown('<div class="section-head">⏱ TIME RANGE</div>', unsafe_allow_html=True)
    if st.session_state.video_meta:
        dur = st.session_state.video_meta["duration_sec"]
        time_range = st.slider("Search window (seconds)", 0.0, float(dur),
                               (0.0, float(dur)), 0.5)
    else:
        time_range = (0.0, 9999.0)
        st.caption("Load a video to set time range.")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sl-header">SMART<span style="color:#cce8f4">LENS</span> <span style="font-size:1rem;color:#3a5568">AI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sl-sub">PROMPT-BASED VIDEO OBJECT FINDER · CLIP EXTRACTION · ENHANCEMENT</div>', unsafe_allow_html=True)

# Load models early
with st.spinner("Initialising AI models..."):
    load_models()

# ── Step 1: Load & Scan Video ─────────────────────────────────────────────────
st.markdown('<div class="section-head">STEP 1 — LOAD & SCAN VIDEO</div>', unsafe_allow_html=True)

col_scan1, col_scan2 = st.columns([2, 1])

with col_scan1:
    if video_source:
        from core.video_loader import load_video_metadata
        if st.session_state.video_path != video_source:
            meta = load_video_metadata(video_source)
            st.session_state.video_path = video_source
            st.session_state.video_meta = meta
            st.session_state.detected_frames = None
            st.session_state.inventory = None

        meta = st.session_state.video_meta
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", meta["duration_str"])
        c2.metric("FPS", f"{meta['fps']:.0f}")
        c3.metric("Resolution", f"{meta['width']}×{meta['height']}")
        c4.metric("Frames", meta["total_frames"])

        if st.button("🔍 Scan Video for Objects", type="primary", use_container_width=True):
            with st.spinner(f"Extracting frames at {sample_fps} fps and running YOLOv8 detection..."):
                from core.video_loader import extract_frames
                from core.detector import detect_all_frames
                from core.inventory import build_inventory

                frames = extract_frames(
                    video_source,
                    sample_fps=sample_fps,
                    start_sec=time_range[0],
                    end_sec=time_range[1],
                )
                detected = detect_all_frames(
                    frames,
                    conf_threshold=conf_threshold,
                    category_filter=cat_filter,
                )
                inventory = build_inventory(detected)

                st.session_state.detected_frames = detected
                st.session_state.inventory = inventory
                st.success(f"Scanned {len(frames)} frames. Found {inventory['summary']['unique_objects']} object types.")
    else:
        st.info("Select a video source in the sidebar to begin.")

with col_scan2:
    if video_source and Path(video_source).exists():
        # Show first frame preview
        cap = cv2.VideoCapture(video_source)
        ret, preview = cap.read()
        cap.release()
        if ret:
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                     caption="Video preview", use_container_width=True)

st.divider()

# ── Step 2: Object Inventory ───────────────────────────────────────────────────
st.markdown('<div class="section-head">STEP 2 — FULL OBJECT INVENTORY</div>', unsafe_allow_html=True)

if st.session_state.inventory:
    inv = st.session_state.inventory
    summary = inv["summary"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Detections",  summary["total_detections"])
    m2.metric("Unique Objects",    summary["unique_objects"])
    m3.metric("Unidentified",      summary["unidentified_count"])

    if inv["objects"]:
        st.markdown("**Detected Objects:**")
        rows_html = ""
        for label, data in inv["objects"].items():
            color = "#00e5ff"
            rows_html += f"""
            <div class="inv-row">
              <span class="inv-label">🏷 {label.upper()}</span>
              <span class="inv-count">{data['count']}× detections</span>
              <span class="inv-conf">conf {data['avg_confidence']:.2f}</span>
              <span class="inv-time">first: {data['first_seen']} → last: {data['last_seen']}</span>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)

    if inv["unidentified"]:
        with st.expander(f"⚠ {summary['unidentified_count']} Unidentified Detections"):
            for u in inv["unidentified"][:10]:
                st.markdown(
                    f'<span class="unid-tag">UNIDENTIFIED</span> '
                    f'<span style="font-family:monospace;font-size:12px;color:#6a8898"> '
                    f'at {u["timestamp_str"]} — conf {u["confidence"]:.2f}</span>',
                    unsafe_allow_html=True,
                )
else:
    st.caption("Scan the video first to see the object inventory.")

st.divider()

# ── Step 3: Prompt Search ──────────────────────────────────────────────────────
st.markdown('<div class="section-head">STEP 3 — PROMPT SEARCH & CLIP EXTRACTION</div>', unsafe_allow_html=True)

# Example prompts
st.markdown("**Try a prompt:**")
example_prompts = [
    "person walking",
    "car moving fast",
    "bicycle",
    "person carrying a bag",
    "truck near the road",
    "two people together",
]
cols_ex = st.columns(len(example_prompts))
for i, ex in enumerate(example_prompts):
    with cols_ex[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["prompt_input"] = ex

# Main prompt input
prompt = st.text_input(
    "Enter your search prompt",
    value=st.session_state.get("prompt_input", ""),
    placeholder="e.g. person carrying a bag near the road / car / find all bicycles ...",
    key="prompt_text",
)

col_b1, col_b2 = st.columns([1, 1])
with col_b1:
    run_prompt = st.button("▶ Search & Extract Clips", type="primary", use_container_width=True,
                           disabled=not (prompt and st.session_state.detected_frames is not None))
with col_b2:
    run_locate = st.button("📍 Locate Object Across Video", use_container_width=True,
                           disabled=not (prompt and st.session_state.detected_frames is not None))

if not st.session_state.detected_frames:
    st.caption("⬆ Scan the video first, then enter a prompt.")

# ── Run Prompt Search ──────────────────────────────────────────────────────────
if run_prompt and prompt and st.session_state.detected_frames:
    from core.clip_matcher import match_prompt_to_frames
    from core.clip_extractor import extract_clips_from_matches
    from utils.response import get_response, get_clip_response
    from utils.visualizer import annotate_match_frame, build_timeline_image

    with st.spinner(f"Running CLIP matching for: \"{prompt}\"..."):
        matches = match_prompt_to_frames(
            prompt,
            st.session_state.detected_frames,
            similarity_threshold=similarity_threshold,
        )

    resp = get_response(prompt, matches, mode="prompt",
                        inventory=st.session_state.inventory)

    # Status badge
    badge_class = {"success": "badge-success", "partial": "badge-warn",
                   "not_found": "badge-error", "error": "badge-error"}.get(resp["status"], "badge-warn")
    st.markdown(f'<div class="{badge_class}">{resp["message"]}</div>', unsafe_allow_html=True)
    if resp["suggestion"]:
        st.caption(resp["suggestion"])

    if matches:
        # Timeline
        dur = st.session_state.video_meta["duration_sec"]
        tl_img = build_timeline_image(matches, dur)
        st.markdown('<div class="tl-wrap"><p style="font-family:monospace;font-size:10px;color:#3a5568;margin:0 0 4px">MATCH TIMELINE</p>', unsafe_allow_html=True)
        st.image(tl_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Extract clips
        output_dir = str(Path(__file__).parent / "outputs")
        with st.spinner("Extracting and enhancing clips..."):
            clips = extract_clips_from_matches(
                st.session_state.video_path,
                matches,
                output_dir=output_dir,
                filters=enhancement_filters,
                padding_sec=1.0,
            )

        clip_resp = get_clip_response(clips, prompt)
        st.markdown(f'<div class="badge-success">{clip_resp["message"]}</div>', unsafe_allow_html=True)

        # Show clips
        st.markdown("### Extracted Clips")
        for i, clip in enumerate(clips):
            with st.expander(
                f"📼 Clip {i+1} — {clip['start_str']} → {clip['end_str']} "
                f"| {clip['match_count']} match(es) | labels: {', '.join(clip['labels'])}",
                expanded=(i == 0),
            ):
                c_left, c_right = st.columns([2, 1])
                with c_left:
                    if Path(clip["clip_path"]).exists():
                        with open(clip["clip_path"], "rb") as f:
                            st.video(f.read())
                    st.markdown(
                        f'<div class="clip-ts">{clip["start_str"]} → {clip["end_str"]}</div>'
                        f'<div class="clip-conf">avg similarity: {clip["avg_similarity"]:.3f} '
                        f'| {clip["match_count"]} detection(s)</div>',
                        unsafe_allow_html=True,
                    )
                with c_right:
                    # Best match frame
                    best = clip["best_match"]
                    vis = annotate_match_frame(best["image"], best, filters=enhancement_filters)
                    st.image(vis["annotated_frame"], caption="Best match frame", use_container_width=True)
                    st.image(vis["zoom_panel"], caption="🔍 Zoom + Enhance", use_container_width=True)

        # Top individual matches
        st.markdown("### Top Individual Matches")
        top_matches = matches[:6]
        cols_m = st.columns(3)
        for i, m in enumerate(top_matches):
            with cols_m[i % 3]:
                vis = annotate_match_frame(m["image"], m, filters=enhancement_filters)
                st.image(vis["annotated_frame"], use_container_width=True)
                st.markdown(
                    f'<div class="clip-ts">⏱ {m["timestamp_str"]}</div>'
                    f'<div class="clip-label">{m["label"]}</div>'
                    f'<div class="clip-conf">CLIP sim: {m["clip_similarity"]:.3f} | '
                    f'conf: {m["confidence"]:.2f}</div>',
                    unsafe_allow_html=True,
                )

# ── Run Object Locator ─────────────────────────────────────────────────────────
if run_locate and prompt and st.session_state.detected_frames:
    from core.clip_matcher import locate_object
    from utils.response import get_response
    from utils.visualizer import annotate_match_frame, build_timeline_image

    with st.spinner(f"Locating \"{prompt}\" across full video..."):
        hits = locate_object(
            prompt,
            st.session_state.detected_frames,
            conf_threshold=conf_threshold,
        )

    resp = get_response(prompt, hits, mode="locate",
                        inventory=st.session_state.inventory)
    badge_class = {"success": "badge-success", "partial": "badge-warn",
                   "not_found": "badge-error"}.get(resp["status"], "badge-warn")
    st.markdown(f'<div class="{badge_class}">{resp["message"]}</div>', unsafe_allow_html=True)
    if resp["suggestion"]:
        st.caption(resp["suggestion"])

    if hits:
        dur = st.session_state.video_meta["duration_sec"]
        tl_img = build_timeline_image(hits, dur)
        st.markdown('<div class="tl-wrap"><p style="font-family:monospace;font-size:10px;color:#3a5568;margin:0 0 4px">LOCATION TIMELINE</p>', unsafe_allow_html=True)
        st.image(tl_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### All Timestamps")
        timestamps = sorted(set(h["timestamp_str"] for h in hits))
        ts_cols = st.columns(4)
        for i, ts in enumerate(timestamps):
            ts_cols[i % 4].markdown(
                f'<div style="font-family:monospace;font-size:13px;color:#00e5ff;'
                f'background:#0d1219;padding:6px 10px;border:1px solid #1a2535;'
                f'border-radius:3px;margin-bottom:4px">⏱ {ts}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Visual Detections")
        cols_h = st.columns(3)
        for i, h in enumerate(hits[:9]):
            with cols_h[i % 3]:
                vis = annotate_match_frame(h["image"], h, filters=enhancement_filters)
                st.image(vis["annotated_frame"], use_container_width=True)
                st.image(vis["zoom_panel"], caption=f"Zoom @ {h['timestamp_str']}", use_container_width=True)

st.divider()
st.markdown(
    '<p style="font-family:monospace;font-size:10px;color:#1a2535;text-align:center">'
    'SMARTLENS AI · CV PROJECT · YOLOv8 + CLIP + OpenCV · Python</p>',
    unsafe_allow_html=True,
)
