"""
exporter.py — Export & Reporting
Generate PDF and JSON reports after a search session.
"""
import json
import os
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
#  JSON EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_json(session_data: dict, output_path: str) -> str:
    """
    Serialize session data (inventory, matches, clips, anomalies) to JSON.
    Returns the output file path.
    """
    def _make_serializable(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return "[frame_image]"
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(i) for i in obj]
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    clean = _make_serializable(session_data)
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_pdf(session_data: dict, output_path: str) -> str:
    """
    Generate a structured PDF report from session data.
    Returns the output file path.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 not installed. Run: pip install fpdf2")

    import cv2
    import tempfile
    import numpy as np

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 180, 220)
    pdf.cell(0, 12, "SmartLens AI — Search Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 120, 140)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(4)

    # ── Video Metadata ──
    meta = session_data.get("video_meta", {})
    if meta:
        _pdf_section(pdf, "VIDEO METADATA")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 60, 80)
        fields = [
            ("File",       meta.get("path", "N/A")),
            ("Duration",   meta.get("duration_str", "N/A")),
            ("Resolution", f"{meta.get('width','?')}×{meta.get('height','?')}"),
            ("FPS",        str(meta.get("fps", "N/A"))),
        ]
        for label, val in fields:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 100, 140)
            pdf.cell(40, 7, label + ":", ln=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 60, 80)
            pdf.cell(0, 7, str(val), ln=True)
        pdf.ln(4)

    # ── Search Query ──
    prompt = session_data.get("prompt", "")
    if prompt:
        _pdf_section(pdf, "SEARCH QUERY")
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(20, 20, 60)
        pdf.multi_cell(0, 7, f'"{prompt}"')
        pdf.ln(4)

    # ── Object Inventory ──
    inventory = session_data.get("inventory", {})
    if inventory and inventory.get("objects"):
        _pdf_section(pdf, "DETECTED OBJECTS (INVENTORY)")
        objects = inventory["objects"]
        summary = inventory.get("summary", {})

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 80, 100)
        pdf.cell(0, 6,
                 f"Total Detections: {summary.get('total_detections',0)}  |  "
                 f"Unique Objects: {summary.get('unique_objects',0)}  |  "
                 f"Unidentified: {summary.get('unidentified_count',0)}", ln=True)
        pdf.ln(2)

        # Table header
        _pdf_table_header(pdf, ["Object", "Count", "Avg Conf", "First Seen", "Last Seen"])
        for label, data in list(objects.items())[:20]:
            cols = [
                label.upper(),
                str(data.get("count", 0)),
                f"{data.get('avg_confidence', 0):.2f}",
                data.get("first_seen", "—"),
                data.get("last_seen", "—"),
            ]
            _pdf_table_row(pdf, cols)
        pdf.ln(4)

    # ── Match Results ──
    matches = session_data.get("matches", [])
    if matches:
        _pdf_section(pdf, f"PROMPT MATCH RESULTS ({len(matches)} matches)")
        _pdf_table_header(pdf, ["Timestamp", "Label", "CLIP Similarity", "Confidence"])
        for m in matches[:30]:
            _pdf_table_row(pdf, [
                m.get("timestamp_str", "—"),
                m.get("label", "—"),
                f"{m.get('clip_similarity', 0):.3f}",
                f"{m.get('confidence', 0):.2f}",
            ])
        if len(matches) > 30:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(120, 140, 160)
            pdf.cell(0, 6, f"  ... and {len(matches)-30} more matches.", ln=True)
        pdf.ln(4)

    # ── Extracted Clips ──
    clips = session_data.get("clips", [])
    if clips:
        _pdf_section(pdf, f"EXTRACTED CLIPS ({len(clips)} segments)")
        for i, clip in enumerate(clips):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 120, 180)
            pdf.cell(0, 7, f"Clip {i+1}: {clip.get('start_str','?')} → {clip.get('end_str','?')}", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 80, 100)
            labels = clip.get("labels", set())
            pdf.cell(0, 5,
                     f"  Labels: {', '.join(labels) if labels else '—'}  |  "
                     f"Matches: {clip.get('match_count',0)}  |  "
                     f"Avg Similarity: {clip.get('avg_similarity', 0):.3f}", ln=True)
            path = clip.get("clip_path", "")
            if path:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(140, 160, 180)
                pdf.cell(0, 5, f"  Saved to: {path}", ln=True)
            pdf.ln(1)

            # Thumbnail
            thumb = clip.get("thumbnail")
            if thumb is not None and isinstance(thumb, np.ndarray):
                try:
                    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    small = cv2.resize(thumb, (160, 90))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    from PIL import Image as PILImage
                    PILImage.fromarray(rgb).save(tmp_img.name)
                    tmp_img.close()
                    pdf.image(tmp_img.name, w=60)
                    os.unlink(tmp_img.name)
                except Exception:
                    pass
            pdf.ln(2)

    # ── Anomaly Report ──
    anomalies = session_data.get("anomalies", [])
    if anomalies:
        _pdf_section(pdf, f"ANOMALY DETECTION ({len(anomalies)} anomalous frames)")
        _pdf_table_header(pdf, ["Timestamp", "Anomaly Score"])
        for a in anomalies[:20]:
            _pdf_table_row(pdf, [
                a.get("timestamp_str", "—"),
                f"{a.get('anomaly_score', 0):.4f}",
            ])
        pdf.ln(4)

    # ── Re-ID Summary ──
    reid_summary = session_data.get("reid_summary", {})
    if reid_summary:
        _pdf_section(pdf, f"PERSON RE-IDENTIFICATION ({len(reid_summary)} unique persons)")
        _pdf_table_header(pdf, ["Person ID", "Appearances", "First Seen", "Last Seen"])
        for pid, info in reid_summary.items():
            _pdf_table_row(pdf, [
                f"Person #{pid}",
                str(info.get("count", 0)),
                info.get("first_seen", "—"),
                info.get("last_seen", "—"),
            ])
        pdf.ln(4)

    # ── Footer ──
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 180, 200)
    pdf.cell(0, 6, "SmartLens AI · YOLOv8 + CLIP + OpenCV · Finals Project", ln=True, align="C")

    os.makedirs(Path(output_path).parent, exist_ok=True)
    pdf.output(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_section(pdf, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 150, 200)
    pdf.set_fill_color(230, 245, 252)
    pdf.cell(0, 8, "  " + title, ln=True, fill=True)
    pdf.ln(2)


def _pdf_table_header(pdf, cols: list):
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(0, 100, 150)
    col_w = 190 // len(cols)
    for col in cols:
        pdf.cell(col_w, 6, col, border=0, fill=True)
    pdf.ln()


def _pdf_table_row(pdf, cols: list, even: bool = False):
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(40, 60, 80)
    pdf.set_fill_color(240, 248, 255)
    col_w = 190 // len(cols)
    for col in cols:
        pdf.cell(col_w, 6, str(col)[:40], border=0, fill=even)
    pdf.ln()


def build_session_data(prompt="", video_meta=None, inventory=None,
                       matches=None, clips=None, anomalies=None,
                       reid_summary=None, tracked=None):
    """Assemble a session data dict for export."""
    return {
        "prompt":       prompt,
        "video_meta":   video_meta or {},
        "inventory":    inventory or {},
        "matches":      matches or [],
        "clips":        clips or [],
        "anomalies":    anomalies or [],
        "reid_summary": reid_summary or {},
        "tracked":      tracked or [],
        "generated_at": datetime.now().isoformat(),
    }
