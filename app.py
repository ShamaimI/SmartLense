"""
app.py — SmartLens AI (Finals Edition)
Enhanced with:
  1. Person Re-Identification (CLIP + FAISS embeddings)
  2. Object Tracking with Trajectories (ByteTrack via Ultralytics)
  3. Scene / Event Description (BLIP-2)
  4. Anomaly Detection (CLIP frame embeddings)
  5. Multi-Video / Batch Search
  6. Export & Reporting (PDF + JSON)
  7. Live Camera / Webcam Support
  8. Spatial Queries (region filters + co-occurrence)

Run:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartLens AI — Finals",
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

section[data-testid="stSidebar"] {
    background: #0a0c11 !important;
    border-right: 1px solid #1a2535 !important;
}

.sl-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.4rem; font-weight: 700;
    color: #00e5ff; letter-spacing: 3px;
    text-shadow: 0 0 30px rgba(0,229,255,0.4);
    margin-bottom: 0;
}
.sl-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; color: #3a5568;
    letter-spacing: 2px; text-transform: uppercase;
    margin-top: 2px; margin-bottom: 20px;
}

.badge-success { background:#0d2a1a; border:1px solid #00ff88; color:#00ff88;
    padding:6px 14px; border-radius:3px; font-family:'Share Tech Mono',monospace;
    font-size:12px; display:inline-block; }
.badge-warn    { background:#2a1e00; border:1px solid #ffaa00; color:#ffaa00;
    padding:6px 14px; border-radius:3px; font-family:'Share Tech Mono',monospace;
    font-size:12px; display:inline-block; }
.badge-error   { background:#2a0d14; border:1px solid #ff3355; color:#ff3355;
    padding:6px 14px; border-radius:3px; font-family:'Share Tech Mono',monospace;
    font-size:12px; display:inline-block; }
.badge-purple  { background:#1a0d2a; border:1px solid #aa66ff; color:#aa66ff;
    padding:6px 14px; border-radius:3px; font-family:'Share Tech Mono',monospace;
    font-size:12px; display:inline-block; }

.clip-card { background:#0d1219; border:1px solid #1a2535;
    border-radius:4px; padding:12px; margin-bottom:10px; }
.clip-ts   { font-family:'Share Tech Mono',monospace; color:#00e5ff; font-size:13px; }
.clip-label { color:#cce8f4; font-size:14px; font-weight:600; }
.clip-conf  { color:#3a5568; font-size:12px; font-family:'Share Tech Mono',monospace; }

.section-head {
    font-family:'Share Tech Mono',monospace; font-size:11px;
    color:#00e5ff; letter-spacing:3px; text-transform:uppercase;
    border-bottom:1px solid #1a2535; padding-bottom:6px; margin-bottom:14px;
}

.inv-row { display:flex; justify-content:space-between; align-items:center;
    padding:8px 0; border-bottom:1px solid #111820; font-size:13px; }
.inv-label { font-weight:600; color:#cce8f4; }
.inv-count { font-family:'Share Tech Mono',monospace; color:#00e5ff; }
.inv-conf  { font-family:'Share Tech Mono',monospace; color:#3a5568; font-size:11px; }
.inv-time  { font-family:'Share Tech Mono',monospace; color:#ffaa00; font-size:11px; }
.unid-tag  { background:#1a0d20; border:1px solid #aa66ff; color:#aa66ff;
    font-size:10px; padding:2px 8px; border-radius:2px;
    font-family:'Share Tech Mono',monospace; }

.tl-wrap { background:#0a0d12; border:1px solid #1a2535; border-radius:3px;
    padding:6px 10px; margin:10px 0; }

.feature-tab { font-family:'Orbitron',sans-serif; font-size:0.7rem; }

.anomaly-card { background:#1a0a0d; border:1px solid #ff3355;
    border-radius:4px; padding:10px; margin-bottom:8px; }
.reid-card { background:#0d0a1a; border:1px solid #aa66ff;
    border-radius:4px; padding:10px; margin-bottom:8px; }
.live-frame { border:2px solid #00e5ff; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "video_path":       None,
        "video_meta":       None,
        "detected_frames":  None,
        "inventory":        None,
        "last_query":       "",
        "last_matches":     [],
        "last_clips":       [],
        "tracked_results":  None,
        "trajectories":     None,
        "anomalies":        None,
        "reid_results":     None,
        "reid_summary":     None,
        "batch_results":    None,
        "live_running":     False,
        "live_records":     [],
        "spatial_results":  None,
        "export_session":   {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sl-header" style="font-size:1.4rem">SMART<span style="color:#fff">LENS</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sl-sub">CV PROJECT · FINALS EDITION</div>', unsafe_allow_html=True)
    st.divider()

    # ── Video Input ──
    st.markdown('<div class="section-head">📁 VIDEO INPUT</div>', unsafe_allow_html=True)
    DATASET_VIDEO = str(Path(__file__).parent / "dataset" / "demo_video.mp4")
    use_dataset = st.checkbox("Use built-in dataset video", value=True)

    if use_dataset:
        video_source = DATASET_VIDEO if Path(DATASET_VIDEO).exists() else None
        if video_source:
            st.success("Dataset video loaded ✓")
        else:
            st.error("Dataset video not found.")
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
    sample_fps         = st.slider("Frame sampling (fps)", 1, 10, 2)
    conf_threshold     = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)
    similarity_threshold = st.slider("CLIP similarity threshold", 0.10, 0.50, 0.20, 0.01)

    CATEGORIES = ["All", "person", "car", "truck", "bus", "bicycle", "motorbike", "bag", "backpack"]
    category   = st.selectbox("Category filter", CATEGORIES)
    cat_filter = None if category == "All" else [category]

    st.divider()

    # ── Enhancement Filters ──
    st.markdown('<div class="section-head">✨ ENHANCEMENT</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        use_clahe   = st.checkbox("CLAHE",   value=False)
        use_denoise = st.checkbox("Denoise", value=False)
    with col2:
        use_deblur  = st.checkbox("Deblur",  value=False)
        use_sharpen = st.checkbox("Sharpen", value=False)
    brightness = st.slider("Brightness", -80, 80, 0)
    contrast   = st.slider("Contrast",   -80, 80, 0)
    enhancement_filters = {
        "clahe": use_clahe, "denoise": use_denoise,
        "deblur": use_deblur, "sharpen": use_sharpen,
        "brightness": brightness, "contrast": contrast,
    }

    st.divider()

    # ── Time Range ──
    st.markdown('<div class="section-head">⏱ TIME RANGE</div>', unsafe_allow_html=True)
    if st.session_state.video_meta:
        dur = st.session_state.video_meta["duration_sec"]
        time_range = st.slider("Search window (s)", 0.0, float(dur),
                               (0.0, float(dur)), 0.5)
    else:
        time_range = (0.0, 9999.0)
        st.caption("Load a video first.")


# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sl-header">SMART<span style="color:#cce8f4">LENS</span> '
    '<span style="font-size:1rem;color:#3a5568">AI · FINALS</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sl-sub">YOLOv8 · CLIP · Re-ID · Tracking · Anomaly · Scene AI · Batch · Live Cam · Spatial Queries · Export</div>',
    unsafe_allow_html=True,
)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🔍 Scan & Search",
    "🧑 Re-ID & Tracking",
    "🚨 Anomaly Detection",
    "🗣 Scene Description",
    "📁 Batch Search",
    "📡 Live Camera",
    "🗺 Spatial Queries",
    "📄 Export & Report",
])

tab_scan, tab_reid, tab_anomaly, tab_scene, tab_batch, tab_live, tab_spatial, tab_export = tabs


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SCAN & SEARCH  (original mids functionality, enhanced)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scan:

    # ── Step 1: Load & Scan ──
    st.markdown('<div class="section-head">STEP 1 — LOAD & SCAN VIDEO</div>', unsafe_allow_html=True)
    col_scan1, col_scan2 = st.columns([2, 1])

    with col_scan1:
        if video_source:
            from core.video_loader import load_video_metadata
            if st.session_state.video_path != video_source:
                meta = load_video_metadata(video_source)
                st.session_state.video_path  = video_source
                st.session_state.video_meta  = meta
                st.session_state.detected_frames = None
                st.session_state.inventory   = None
                st.session_state.tracked_results = None
                st.session_state.anomalies   = None

            meta = st.session_state.video_meta
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration",   meta["duration_str"])
            c2.metric("FPS",        f"{meta['fps']:.0f}")
            c3.metric("Resolution", f"{meta['width']}×{meta['height']}")
            c4.metric("Frames",     meta["total_frames"])

            if st.button("🔍 Scan Video for Objects", type="primary", use_container_width=True):
                with st.spinner("Extracting frames & running YOLOv8..."):
                    from core.video_loader import extract_frames
                    from core.detector import detect_all_frames
                    from core.inventory import build_inventory

                    frames   = extract_frames(video_source, sample_fps=sample_fps,
                                             start_sec=time_range[0], end_sec=time_range[1])
                    detected = detect_all_frames(frames, conf_threshold=conf_threshold,
                                                 category_filter=cat_filter)
                    inventory = build_inventory(detected)

                    st.session_state.detected_frames = detected
                    st.session_state.inventory       = inventory
                    st.success(f"Scanned {len(frames)} frames. Found {inventory['summary']['unique_objects']} object types.")
        else:
            st.info("Select a video source in the sidebar.")

    with col_scan2:
        if video_source and Path(video_source).exists():
            cap = cv2.VideoCapture(video_source)
            ret, preview = cap.read()
            cap.release()
            if ret:
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                         caption="Video preview", use_container_width=True)

    st.divider()

    # ── Step 2: Inventory ──
    st.markdown('<div class="section-head">STEP 2 — FULL OBJECT INVENTORY</div>', unsafe_allow_html=True)

    if st.session_state.inventory:
        inv     = st.session_state.inventory
        summary = inv["summary"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Detections", summary["total_detections"])
        m2.metric("Unique Objects",   summary["unique_objects"])
        m3.metric("Unidentified",     summary["unidentified_count"])

        if inv["objects"]:
            rows_html = ""
            for label, data in inv["objects"].items():
                rows_html += f"""
                <div class="inv-row">
                  <span class="inv-label">🏷 {label.upper()}</span>
                  <span class="inv-count">{data['count']}×</span>
                  <span class="inv-conf">conf {data['avg_confidence']:.2f}</span>
                  <span class="inv-time">{data['first_seen']} → {data['last_seen']}</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)

        if inv["unidentified"]:
            with st.expander(f"⚠ {summary['unidentified_count']} Unidentified Detections"):
                for u in inv["unidentified"][:10]:
                    st.markdown(
                        f'<span class="unid-tag">UNIDENTIFIED</span> '
                        f'<span style="font-family:monospace;font-size:12px;color:#6a8898">'
                        f'at {u["timestamp_str"]} — conf {u["confidence"]:.2f}</span>',
                        unsafe_allow_html=True,
                    )
    else:
        st.caption("Scan the video first to see the inventory.")

    st.divider()

    # ── Step 3: Prompt Search ──
    st.markdown('<div class="section-head">STEP 3 — PROMPT SEARCH & CLIP EXTRACTION</div>', unsafe_allow_html=True)

    example_prompts = ["person walking", "car moving fast", "bicycle",
                       "person carrying a bag", "truck near the road", "two people together"]
    cols_ex = st.columns(len(example_prompts))
    for i, ex in enumerate(example_prompts):
        with cols_ex[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["prompt_input"] = ex

    prompt = st.text_input(
        "Enter search prompt",
        value=st.session_state.get("prompt_input", ""),
        placeholder="e.g. person carrying a bag / car / bicycle near the road ...",
        key="prompt_text",
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        run_prompt = st.button("▶ Search & Extract Clips", type="primary",
                               use_container_width=True,
                               disabled=not (prompt and st.session_state.detected_frames is not None))
    with col_b2:
        run_locate = st.button("📍 Locate Object Across Video", use_container_width=True,
                               disabled=not (prompt and st.session_state.detected_frames is not None))

    # ── Run Search ──
    if run_prompt and prompt and st.session_state.detected_frames:
        from core.clip_matcher import match_prompt_to_frames
        from core.clip_extractor import extract_clips_from_matches
        from utils.response import get_response, get_clip_response
        from utils.visualizer import annotate_match_frame, build_timeline_image

        with st.spinner(f'Matching "{prompt}"...'):
            matches = match_prompt_to_frames(
                prompt, st.session_state.detected_frames,
                similarity_threshold=similarity_threshold,
            )
        st.session_state.last_matches = matches
        st.session_state.last_query   = prompt

        resp        = get_response(prompt, matches, mode="prompt", inventory=st.session_state.inventory)
        badge_class = {"success":"badge-success","partial":"badge-warn",
                       "not_found":"badge-error","error":"badge-error"}.get(resp["status"],"badge-warn")
        st.markdown(f'<div class="{badge_class}">{resp["message"]}</div>', unsafe_allow_html=True)
        if resp["suggestion"]:
            st.caption(resp["suggestion"])

        if matches:
            dur    = st.session_state.video_meta["duration_sec"]
            tl_img = build_timeline_image(matches, dur)
            st.markdown('<div class="tl-wrap"><p style="font-family:monospace;font-size:10px;color:#3a5568;margin:0 0 4px">MATCH TIMELINE</p>', unsafe_allow_html=True)
            st.image(tl_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            output_dir = str(Path(__file__).parent / "outputs")
            with st.spinner("Extracting clips..."):
                clips = extract_clips_from_matches(
                    st.session_state.video_path, matches, output_dir=output_dir,
                    filters=enhancement_filters, padding_sec=1.0,
                )
            st.session_state.last_clips = clips
            # Store for export
            st.session_state.export_session.update({
                "prompt": prompt, "matches": matches, "clips": clips,
                "video_meta": st.session_state.video_meta,
                "inventory": st.session_state.inventory,
            })

            clip_resp = get_clip_response(clips, prompt)
            st.markdown(f'<div class="badge-success">{clip_resp["message"]}</div>', unsafe_allow_html=True)

            st.markdown("### Extracted Clips")
            for i, clip in enumerate(clips):
                with st.expander(
                    f"📼 Clip {i+1} — {clip['start_str']} → {clip['end_str']} "
                    f"| {clip['match_count']} match(es) | {', '.join(clip['labels'])}",
                    expanded=(i == 0),
                ):
                    c_left, c_right = st.columns([2, 1])
                    with c_left:
                        if Path(clip["clip_path"]).exists():
                            with open(clip["clip_path"], "rb") as f:
                                st.video(f.read())
                        st.markdown(
                            f'<div class="clip-ts">{clip["start_str"]} → {clip["end_str"]}</div>'
                            f'<div class="clip-conf">avg sim: {clip["avg_similarity"]:.3f} | {clip["match_count"]} det(s)</div>',
                            unsafe_allow_html=True,
                        )
                    with c_right:
                        best = clip["best_match"]
                        vis  = annotate_match_frame(best["image"], best, filters=enhancement_filters)
                        st.image(vis["annotated_frame"], caption="Best match", use_container_width=True)
                        st.image(vis["zoom_panel"], caption="🔍 Zoom + Enhance", use_container_width=True)

            st.markdown("### Top Matches")
            cols_m = st.columns(3)
            for i, m in enumerate(matches[:6]):
                with cols_m[i % 3]:
                    vis = annotate_match_frame(m["image"], m, filters=enhancement_filters)
                    st.image(vis["annotated_frame"], use_container_width=True)
                    st.markdown(
                        f'<div class="clip-ts">⏱ {m["timestamp_str"]}</div>'
                        f'<div class="clip-label">{m["label"]}</div>'
                        f'<div class="clip-conf">CLIP: {m["clip_similarity"]:.3f} | conf: {m["confidence"]:.2f}</div>',
                        unsafe_allow_html=True,
                    )

    # ── Locate ──
    if run_locate and prompt and st.session_state.detected_frames:
        from core.clip_matcher import locate_object
        from utils.response import get_response
        from utils.visualizer import annotate_match_frame, build_timeline_image

        with st.spinner(f'Locating "{prompt}"...'):
            hits = locate_object(prompt, st.session_state.detected_frames, conf_threshold=conf_threshold)

        resp        = get_response(prompt, hits, mode="locate", inventory=st.session_state.inventory)
        badge_class = {"success":"badge-success","partial":"badge-warn",
                       "not_found":"badge-error"}.get(resp["status"],"badge-warn")
        st.markdown(f'<div class="{badge_class}">{resp["message"]}</div>', unsafe_allow_html=True)

        if hits:
            dur    = st.session_state.video_meta["duration_sec"]
            tl_img = build_timeline_image(hits, dur)
            st.markdown('<div class="tl-wrap"><p style="font-family:monospace;font-size:10px;color:#3a5568;margin:0 0 4px">LOCATION TIMELINE</p>', unsafe_allow_html=True)
            st.image(tl_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            timestamps = sorted(set(h["timestamp_str"] for h in hits))
            ts_cols = st.columns(4)
            for i, ts in enumerate(timestamps):
                ts_cols[i % 4].markdown(
                    f'<div style="font-family:monospace;font-size:13px;color:#00e5ff;'
                    f'background:#0d1219;padding:6px 10px;border:1px solid #1a2535;'
                    f'border-radius:3px;margin-bottom:4px">⏱ {ts}</div>',
                    unsafe_allow_html=True,
                )

            cols_h = st.columns(3)
            for i, h in enumerate(hits[:9]):
                with cols_h[i % 3]:
                    vis = annotate_match_frame(h["image"], h, filters=enhancement_filters)
                    st.image(vis["annotated_frame"], use_container_width=True)
                    st.image(vis["zoom_panel"], caption=f"@ {h['timestamp_str']}", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PERSON RE-ID & OBJECT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reid:
    st.markdown('<div class="section-head">PERSON RE-IDENTIFICATION & OBJECT TRACKING</div>', unsafe_allow_html=True)

    if not st.session_state.video_path:
        st.info("Load a video in the sidebar and scan it first (Tab 1).")
    else:
        col_reid1, col_reid2 = st.columns(2)

        # ── Re-ID ──────────────────────────────────────────────────────────
        with col_reid1:
            st.markdown("#### 🧑 Person Re-Identification")
            st.caption("Assigns persistent IDs to persons across frames using CLIP embeddings.")
            reid_threshold = st.slider("Re-ID similarity threshold", 0.60, 0.99, 0.82, 0.01, key="reid_thr")

            if st.button("Run Person Re-ID", type="primary", use_container_width=True,
                         disabled=st.session_state.detected_frames is None):
                from core.reid import PersonReID

                with st.spinner("Running Re-ID across all detected frames..."):
                    reid = PersonReID(similarity_threshold=reid_threshold)

                    # Flatten detections
                    flat = []
                    for fd in st.session_state.detected_frames:
                        for det in fd.get("detections", []):
                            if det["label"] == "person":
                                flat.append({**det,
                                             "image":         fd["image"],
                                             "timestamp_str": fd["timestamp_str"],
                                             "timestamp_sec": fd["timestamp_sec"]})

                    results = reid.process_detections(flat)
                    summary = reid.get_person_summary()

                st.session_state.reid_results = results
                st.session_state.reid_summary = summary
                st.session_state.export_session["reid_summary"] = summary
                st.success(f"Found {len(summary)} unique person(s) across the video.")

            if st.session_state.reid_summary:
                summary = st.session_state.reid_summary
                for pid, info in summary.items():
                    with st.expander(f"Person #{pid} — {info['count']} appearances"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Appearances", info["count"])
                        c2.metric("First Seen",  info["first_seen"])
                        c3.metric("Last Seen",   info["last_seen"])
                        if info["timestamps"]:
                            st.markdown("**All timestamps:**")
                            st.code(" · ".join(info["timestamps"][:20]))

                # Mosaic
                from core.reid import PersonReID as _R
                reid_inst = _R()
                reid_inst.person_tracks = {}
                if st.session_state.reid_results:
                    from collections import defaultdict
                    tracks = defaultdict(list)
                    for d in st.session_state.reid_results:
                        if d.get("person_id") is not None:
                            tracks[d["person_id"]].append(d)
                    from utils.visualizer import make_reid_mosaic
                    mosaic = make_reid_mosaic(dict(tracks))
                    if mosaic:
                        st.image(mosaic, caption="Person thumbnails", use_container_width=True)

        # ── Tracking ───────────────────────────────────────────────────────
        with col_reid2:
            st.markdown("#### 📍 Object Tracking (ByteTrack)")
            st.caption("Tracks objects with persistent IDs and draws trajectory paths.")

            if st.button("Run Object Tracking", type="primary", use_container_width=True):
                from core.tracker import ObjectTracker
                from utils.visualizer import draw_trajectories

                with st.spinner("Running ByteTrack on video..."):
                    try:
                        tracker = ObjectTracker()
                        tracked = tracker.track_video(
                            st.session_state.video_path,
                            sample_fps=sample_fps,
                            conf_threshold=conf_threshold,
                            category_filter=cat_filter,
                            start_sec=time_range[0],
                            end_sec=time_range[1],
                        )
                        trajectories = tracker.get_trajectories()
                        unique_tracks = tracker.get_unique_tracks(tracked)

                        st.session_state.tracked_results = tracked
                        st.session_state.trajectories    = trajectories

                        st.success(f"Tracked {len(unique_tracks)} unique object(s).")

                    except Exception as e:
                        st.error(f"Tracking error: {e}")
                        st.caption("ByteTrack requires YOLOv8 with tracking support.")

            if st.session_state.tracked_results:
                tracks_data = st.session_state.tracked_results
                trajs       = st.session_state.trajectories or {}

                unique_tracks = {}
                from collections import defaultdict
                td = defaultdict(list)
                for d in tracks_data:
                    td[d["track_id"]].append(d)
                unique_tracks = dict(td)

                st.markdown(f"**{len(unique_tracks)} tracked objects:**")
                for tid, dets in list(unique_tracks.items())[:8]:
                    label = dets[0]["label"]
                    st.markdown(
                        f'<span class="clip-ts">Track #{tid}</span> '
                        f'<span class="clip-label"> {label.upper()}</span> '
                        f'<span class="clip-conf"> ({len(dets)} frames, '
                        f'{dets[0]["timestamp_str"]} → {dets[-1]["timestamp_str"]})</span>',
                        unsafe_allow_html=True,
                    )

                # Draw trajectories on a representative frame
                if trajs and tracks_data:
                    from utils.visualizer import draw_trajectories
                    sample_frame = tracks_data[-1]["image"].copy()
                    traj_img = draw_trajectories(sample_frame, trajs)
                    st.image(cv2.cvtColor(traj_img, cv2.COLOR_BGR2RGB),
                             caption="Trajectory Map (last frame)", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<div class="section-head">ANOMALY DETECTION — CLIP EMBEDDING OUTLIERS</div>', unsafe_allow_html=True)
    st.caption("Uses CLIP frame embeddings to identify frames that deviate statistically from the rest of the video. Useful for CCTV and unusual event detection.")

    if st.session_state.detected_frames is None:
        st.info("Scan the video first (Tab 1 → Scan Video).")
    else:
        percentile = st.slider("Anomaly percentile threshold", 70, 99, 90, 1,
                               help="Top N% most unusual frames are flagged.")

        if st.button("🚨 Detect Anomalies", type="primary", use_container_width=True):
            from core.anomaly_detector import detect_anomalies, get_anomaly_summary

            with st.spinner("Computing CLIP embeddings and anomaly scores..."):
                anomalies = detect_anomalies(
                    st.session_state.detected_frames,
                    threshold_percentile=percentile,
                )
                anom_summary = get_anomaly_summary(anomalies)

            st.session_state.anomalies = anomalies
            st.session_state.export_session["anomalies"] = anomalies

            if anomalies:
                st.markdown(f'<div class="badge-warn">⚠ {anom_summary["count"]} anomalous frame(s) detected</div>', unsafe_allow_html=True)
                a1, a2, a3 = st.columns(3)
                a1.metric("Anomalous Frames", anom_summary["count"])
                a2.metric("Max Score",        f'{anom_summary["max_score"]:.4f}')
                a3.metric("Min Score",        f'{anom_summary["min_score"]:.4f}')

                st.markdown("**Top Anomalous Timestamps:**")
                for ts in anom_summary["top_timestamps"]:
                    st.markdown(
                        f'<span style="font-family:monospace;color:#ff3355">🔴 {ts}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No significant anomalies detected.")

        if st.session_state.anomalies:
            st.divider()
            st.markdown("### Anomalous Frames")
            cols_a = st.columns(3)
            for i, anom in enumerate(st.session_state.anomalies[:9]):
                with cols_a[i % 3]:
                    rgb = cv2.cvtColor(anom["image"], cv2.COLOR_BGR2RGB)
                    st.image(rgb, use_container_width=True)
                    st.markdown(
                        f'<div class="anomaly-card">'
                        f'<span class="clip-ts">⏱ {anom["timestamp_str"]}</span><br>'
                        f'<span style="color:#ff3355;font-family:monospace;font-size:11px">'
                        f'anomaly score: {anom["anomaly_score"]:.4f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SCENE DESCRIPTION (BLIP-2)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scene:
    st.markdown('<div class="section-head">SCENE & EVENT DESCRIPTION — BLIP-2 VISION-LANGUAGE MODEL</div>', unsafe_allow_html=True)
    st.caption("Auto-generates natural language descriptions of frames or extracted clips using BLIP-2.")

    st.warning("⚠ BLIP-2 downloads ~2GB on first use. Requires GPU for practical speed. CPU inference is slow but works.")

    if st.session_state.detected_frames is None and not st.session_state.last_clips:
        st.info("Scan the video first (Tab 1) or extract some clips.")
    else:
        scene_mode = st.radio("Describe:", ["Key Frames from Video", "Extracted Clips"], horizontal=True)
        question = st.text_input("Optional: Ask a specific question about the scene",
                                 placeholder="e.g. How many people are in the frame?")

        if st.button("🗣 Generate Scene Descriptions", type="primary"):
            from core.scene_describer import describe_frame, describe_clip

            if scene_mode == "Key Frames from Video" and st.session_state.detected_frames:
                # Pick frames with most detections
                sorted_frames = sorted(
                    st.session_state.detected_frames,
                    key=lambda x: len(x.get("detections", [])),
                    reverse=True,
                )
                top_frames = sorted_frames[:5]

                st.markdown("### Frame Descriptions")
                for fd in top_frames:
                    with st.spinner(f"Describing frame at {fd['timestamp_str']}..."):
                        try:
                            desc = describe_frame(fd["image"], question=question or None)
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(cv2.cvtColor(fd["image"], cv2.COLOR_BGR2RGB),
                                         use_container_width=True)
                            with col2:
                                st.markdown(f'<span class="clip-ts">⏱ {fd["timestamp_str"]}</span>', unsafe_allow_html=True)
                                st.markdown(f"**{desc}**")
                        except Exception as e:
                            st.error(f"BLIP-2 error: {e}")
                            break

            elif scene_mode == "Extracted Clips" and st.session_state.last_clips:
                for clip in st.session_state.last_clips[:3]:
                    with st.spinner(f"Describing clip {clip['start_str']} → {clip['end_str']}..."):
                        try:
                            result = describe_clip(clip["clip_path"], num_keyframes=4,
                                                   question=question or None)
                            st.markdown(f"**Clip {clip['start_str']} → {clip['end_str']}**")
                            st.info(result["summary"])
                            for kf in result["descriptions"]:
                                st.markdown(f"- Frame {kf['keyframe_index']}: {kf['description']}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.warning("No data available for the selected mode.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — BATCH MULTI-VIDEO SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-head">MULTI-VIDEO BATCH SEARCH</div>', unsafe_allow_html=True)
    st.caption("Upload multiple videos and search across all of them simultaneously — simulates a multi-camera surveillance setup.")

    batch_uploads = st.file_uploader(
        "Upload multiple videos",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    batch_prompt = st.text_input("Batch search prompt", placeholder="e.g. person with bag / car",
                                  key="batch_prompt")

    if st.button("▶ Run Batch Search", type="primary",
                 disabled=not (batch_uploads and batch_prompt),
                 use_container_width=True):
        from core.batch_processor import batch_process_videos, batch_search, get_batch_summary

        # Save uploads to temp files
        tmp_paths = []
        for up in batch_uploads:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tf.write(up.read())
            tf.flush()
            tmp_paths.append((tf.name, up.name))

        video_paths = [p[0] for p in tmp_paths]
        video_names = {p[0]: p[1] for p in tmp_paths}

        progress_bar = st.progress(0)
        status_text  = st.empty()

        def progress_cb(i, total, name):
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Processing {name} ({i+1}/{total})...")

        with st.spinner("Processing all videos..."):
            batch_results = batch_process_videos(
                video_paths, sample_fps=sample_fps,
                conf_threshold=conf_threshold, category_filter=cat_filter,
                progress_callback=progress_cb,
            )

        with st.spinner(f'Searching for "{batch_prompt}" across all videos...'):
            search_results = batch_search(batch_prompt, batch_results,
                                          similarity_threshold=similarity_threshold)

        st.session_state.batch_results = search_results
        progress_bar.empty()
        status_text.empty()

        summary = get_batch_summary(search_results)
        s1, s2, s3 = st.columns(3)
        s1.metric("Videos Processed",     summary["total_videos"])
        s2.metric("Videos with Matches",  summary["videos_with_matches"])
        s3.metric("Total Matches",        summary["total_matches"])

    if st.session_state.batch_results:
        st.divider()
        st.markdown("### Batch Search Results")
        for res in st.session_state.batch_results:
            vname = res["video_name"]
            count = res.get("match_count", 0)
            err   = res.get("error")
            label = f"📹 {vname} — {count} match(es)" if not err else f"❌ {vname} — ERROR"

            with st.expander(label, expanded=(count > 0)):
                if err:
                    st.error(f"Error: {err}")
                    continue
                if not res["matches"]:
                    st.caption("No matches found in this video.")
                    continue
                from utils.visualizer import build_timeline_image
                # Fake duration from last timestamp
                dur = max(m["timestamp_sec"] for m in res["matches"]) + 5
                tl  = build_timeline_image(res["matches"], dur)
                st.image(tl, use_container_width=True)

                cols_bm = st.columns(4)
                for i, m in enumerate(res["matches"][:8]):
                    with cols_bm[i % 4]:
                        rgb = cv2.cvtColor(m["image"], cv2.COLOR_BGR2RGB)
                        st.image(rgb, use_container_width=True)
                        st.caption(f'{m["timestamp_str"]} | sim:{m["clip_similarity"]:.2f}')


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — LIVE CAMERA / WEBCAM
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="section-head">LIVE CAMERA / WEBCAM DETECTION</div>', unsafe_allow_html=True)
    st.caption("Real-time YOLOv8 inference on webcam or IP camera stream.")

    from core.live_camera import LiveCamera, list_available_cameras

    col_live_l, col_live_r = st.columns([2, 1])

    with col_live_r:
        st.markdown("#### Camera Settings")
        camera_source_type = st.radio("Source", ["Webcam", "RTSP/URL"], horizontal=True)

        if camera_source_type == "Webcam":
            cam_idx = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
            cam_source = int(cam_idx)
        else:
            cam_source = st.text_input("RTSP URL", placeholder="rtsp://user:pass@ip:port/stream")

        live_conf   = st.slider("Live confidence threshold", 0.1, 0.9, 0.4, 0.05, key="live_conf")
        num_frames  = st.slider("Capture N frames", 1, 60, 10, key="live_frames")
        record_live = st.checkbox("Save captured frames for later analysis", value=True)

        st.markdown("#### Controls")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            btn_capture = st.button("📸 Capture Frames", type="primary", use_container_width=True)
        with col_l2:
            btn_clear = st.button("🗑 Clear", use_container_width=True)

    with col_live_l:
        st.markdown("#### Live Feed")

        if btn_clear:
            st.session_state.live_records = []
            st.info("Cleared.")

        if btn_capture:
            cam = LiveCamera(source=cam_source, conf_threshold=live_conf)
            if cam.open():
                progress_bar = st.progress(0)
                frame_slot   = st.empty()
                records      = []

                for i in range(num_frames):
                    annotated, dets = cam.snapshot()
                    if annotated is None:
                        st.warning("Failed to read frame.")
                        break
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_slot.image(rgb, caption=f"Frame {i+1}/{num_frames} | FPS: {cam.get_fps():.1f}",
                                     use_container_width=True)
                    progress_bar.progress((i + 1) / num_frames)

                    if record_live:
                        from core.live_camera import build_live_detection_record
                        import time
                        rec = build_live_detection_record(
                            annotated.copy(), dets, i, time.time()
                        )
                        records.append(rec)

                cam.release()
                progress_bar.empty()

                if record_live and records:
                    st.session_state.live_records = records
                    st.success(f"Captured {len(records)} frames with detections. You can now run analysis on these in Tab 1 (set detected_frames manually).")
            else:
                st.error(f"Could not open camera source: {cam_source}")
                st.caption("Make sure your webcam is connected and not in use by another application.")

        # Show stored live records
        if st.session_state.live_records:
            st.markdown(f"**{len(st.session_state.live_records)} captured frames:**")
            cols_lr = st.columns(4)
            for i, rec in enumerate(st.session_state.live_records[:8]):
                with cols_lr[i % 4]:
                    rgb = cv2.cvtColor(rec["image"], cv2.COLOR_BGR2RGB)
                    st.image(rgb, use_container_width=True)
                    n_det = len(rec.get("detections", []))
                    st.caption(f"{n_det} detection(s)")

            if st.button("📊 Use Live Frames as Detection Source"):
                st.session_state.detected_frames = st.session_state.live_records
                from core.inventory import build_inventory
                st.session_state.inventory = build_inventory(st.session_state.live_records)
                st.success("Live frames loaded! Switch to Tab 1 to search them.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — SPATIAL QUERIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_spatial:
    st.markdown('<div class="section-head">SPATIAL QUERIES — REGION FILTERS & CO-OCCURRENCE</div>', unsafe_allow_html=True)
    st.caption("Filter detections by screen region or find two objects appearing close together in the same frame.")

    if st.session_state.detected_frames is None:
        st.info("Scan the video first (Tab 1).")
    else:
        from core.spatial_filter import filter_by_region, find_co_occurring_objects, REGIONS

        # Flatten detections for spatial queries
        def flatten_detections(detected_frames):
            flat = []
            for fd in detected_frames:
                for det in fd.get("detections", []):
                    flat.append({
                        **det,
                        "image":         fd["image"],
                        "timestamp_sec": fd["timestamp_sec"],
                        "timestamp_str": fd["timestamp_str"],
                        "frame_idx":     fd["frame_idx"],
                        "clip_similarity": 1.0,
                    })
            return flat

        col_sp1, col_sp2 = st.columns(2)

        # ── Region Filter ──
        with col_sp1:
            st.markdown("#### 🗺 Region Filter")
            region_names = ["Any"] + list(REGIONS.keys())
            selected_region = st.selectbox("Screen region", region_names, key="region_sel")
            region_label_filter = st.selectbox(
                "Filter by object type (optional)",
                ["All"] + ["person", "car", "truck", "bus", "bicycle", "motorbike", "bag"],
                key="region_label",
            )

            if st.button("Apply Region Filter", use_container_width=True):
                flat = flatten_detections(st.session_state.detected_frames)
                filtered = filter_by_region(flat, selected_region if selected_region != "Any" else None)
                if region_label_filter != "All":
                    filtered = [d for d in filtered if d["label"] == region_label_filter]

                st.session_state.spatial_results = filtered
                st.markdown(f'<div class="badge-success">{len(filtered)} detection(s) in region "{selected_region}"</div>', unsafe_allow_html=True)

            if st.session_state.spatial_results:
                results = st.session_state.spatial_results
                cols_sp = st.columns(3)
                for i, det in enumerate(results[:9]):
                    with cols_sp[i % 3]:
                        from utils.visualizer import annotate_match_frame
                        vis = annotate_match_frame(det["image"], det)
                        st.image(vis["annotated_frame"], use_container_width=True)
                        st.caption(f'{det["timestamp_str"]} | {det["label"]}')

        # ── Co-occurrence ──
        with col_sp2:
            st.markdown("#### 🔗 Object Co-occurrence")
            st.caption("Find frames where two object types appear close together.")

            available_labels = ["person", "car", "truck", "bus", "bicycle", "motorbike", "bag", "backpack"]
            obj_a = st.selectbox("Object A", available_labels, index=0, key="cooc_a")
            obj_b = st.selectbox("Object B", available_labels, index=2, key="cooc_b")
            max_dist = st.slider("Max distance between objects (px)", 50, 600, 200, 25, key="cooc_dist")

            if st.button("Find Co-occurring Objects", use_container_width=True):
                flat = flatten_detections(st.session_state.detected_frames)
                co_results = find_co_occurring_objects(flat, obj_a, obj_b, max_distance_px=max_dist)

                if co_results:
                    st.markdown(f'<div class="badge-success">{len(co_results)} co-occurrence(s) of {obj_a} + {obj_b}</div>', unsafe_allow_html=True)
                    cols_co = st.columns(3)
                    for i, res in enumerate(co_results[:9]):
                        with cols_co[i % 3]:
                            from utils.visualizer import annotate_match_frame
                            vis = annotate_match_frame(res["image"], res["object_a"])
                            st.image(vis["annotated_frame"], use_container_width=True)
                            st.caption(
                                f'{res["timestamp_str"]} | dist: {res["distance_px"]:.0f}px'
                            )
                else:
                    st.warning(f'No frames where "{obj_a}" and "{obj_b}" appear within {max_dist}px of each other.')

        # ── Region Visualizer ──
        st.divider()
        st.markdown("#### Region Map Reference")
        reg_img = np.zeros((270, 480, 3), dtype=np.uint8)
        reg_img[:] = (13, 18, 25)
        REGION_DRAW = {
            "top-left":     (0, 0, 240, 135),
            "top-right":    (240, 0, 480, 135),
            "bottom-left":  (0, 135, 240, 270),
            "bottom-right": (240, 135, 480, 270),
            "center":       (120, 67, 360, 202),
        }
        REGION_COLORS = [
            (0, 229, 255), (0, 255, 128), (255, 165, 0),
            (180, 0, 255), (255, 255, 0),
        ]
        for (rname, (x1, y1, x2, y2)), color in zip(REGION_DRAW.items(), REGION_COLORS):
            overlay = reg_img.copy()
            cv2.rectangle(overlay, (x1+2, y1+2), (x2-2, y2-2), color, 1)
            cv2.addWeighted(overlay, 0.15, reg_img, 0.85, 0, reg_img)
            cv2.rectangle(reg_img, (x1+2, y1+2), (x2-2, y2-2), color, 1)
            cv2.putText(reg_img, rname, (x1+6, (y1+y2)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        st.image(cv2.cvtColor(reg_img, cv2.COLOR_BGR2RGB), caption="Screen regions used by spatial filter")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 8 — EXPORT & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown('<div class="section-head">EXPORT & REPORTING — PDF & JSON</div>', unsafe_allow_html=True)
    st.caption("Download a full session report including inventory, matches, clips, anomalies, and Re-ID results.")

    session = st.session_state.export_session
    inv     = st.session_state.inventory
    matches = st.session_state.last_matches
    clips   = st.session_state.last_clips
    anomalies = st.session_state.anomalies or []
    reid_sum  = st.session_state.reid_summary or {}

    # Summary of what's available
    st.markdown("#### Session Data Available for Export")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Inventory",   "✓" if inv else "—")
    c2.metric("Matches",     len(matches) if matches else "—")
    c3.metric("Clips",       len(clips)   if clips   else "—")
    c4.metric("Anomalies",   len(anomalies) if anomalies else "—")
    c5.metric("Re-ID",       len(reid_sum)  if reid_sum  else "—")

    if not any([inv, matches, clips, anomalies, reid_sum]):
        st.warning("Run some analysis first (scan video, search, detect anomalies, run Re-ID) to populate the report.")
    else:
        col_e1, col_e2 = st.columns(2)
        output_dir = str(Path(__file__).parent / "outputs")
        os.makedirs(output_dir, exist_ok=True)

        with col_e1:
            st.markdown("#### 📄 PDF Report")
            if st.button("Generate PDF Report", type="primary", use_container_width=True):
                from core.exporter import export_pdf, build_session_data
                from datetime import datetime

                with st.spinner("Generating PDF..."):
                    sdata = build_session_data(
                        prompt      = session.get("prompt", ""),
                        video_meta  = st.session_state.video_meta,
                        inventory   = inv,
                        matches     = matches,
                        clips       = clips,
                        anomalies   = anomalies,
                        reid_summary = reid_sum,
                    )
                    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_pdf = os.path.join(output_dir, f"smartlens_report_{ts}.pdf")
                    try:
                        export_pdf(sdata, out_pdf)
                        with open(out_pdf, "rb") as f:
                            st.download_button(
                                "⬇ Download PDF Report",
                                data=f.read(),
                                file_name=f"smartlens_report_{ts}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        st.success("PDF generated!")
                    except ImportError:
                        st.error("fpdf2 not installed. Run: pip install fpdf2")
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")

        with col_e2:
            st.markdown("#### 📦 JSON Export")
            if st.button("Generate JSON Export", type="primary", use_container_width=True):
                from core.exporter import export_json, build_session_data
                from datetime import datetime

                with st.spinner("Generating JSON..."):
                    sdata = build_session_data(
                        prompt       = session.get("prompt", ""),
                        video_meta   = st.session_state.video_meta,
                        inventory    = inv,
                        matches      = matches,
                        clips        = clips,
                        anomalies    = anomalies,
                        reid_summary = reid_sum,
                    )
                    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_json = os.path.join(output_dir, f"smartlens_report_{ts}.json")
                    try:
                        export_json(sdata, out_json)
                        with open(out_json, "rb") as f:
                            st.download_button(
                                "⬇ Download JSON Export",
                                data=f.read(),
                                file_name=f"smartlens_report_{ts}.json",
                                mime="application/json",
                                use_container_width=True,
                            )
                        st.success("JSON exported!")
                    except Exception as e:
                        st.error(f"JSON export error: {e}")

        st.divider()
        st.markdown("#### Quick Preview")
        if inv and inv.get("objects"):
            st.markdown("**Detected Objects:**")
            obj_list = [(k, v["count"], v["avg_confidence"])
                        for k, v in inv["objects"].items()]
            for label, count, conf in obj_list[:10]:
                st.markdown(
                    f'<div class="inv-row"><span class="inv-label">{label.upper()}</span>'
                    f'<span class="inv-count">{count}×</span>'
                    f'<span class="inv-conf">conf {conf:.2f}</span></div>',
                    unsafe_allow_html=True,
                )

        if matches:
            st.markdown(f"**{len(matches)} matches** for prompt: *\"{session.get('prompt','')}\"*")
        if anomalies:
            st.markdown(f"**{len(anomalies)} anomalous frames** detected.")
        if reid_sum:
            st.markdown(f"**{len(reid_sum)} unique persons** identified via Re-ID.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="font-family:monospace;font-size:10px;color:#1a2535;text-align:center">'
    'SMARTLENS AI · FINALS EDITION · YOLOv8 + CLIP + BLIP-2 + ByteTrack + FAISS + OpenCV · Python 3.10+</p>',
    unsafe_allow_html=True,
)
