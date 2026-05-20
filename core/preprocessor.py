"""
preprocessor.py
All preprocessing and enhancement filters applied to frames.
"""
import cv2
import numpy as np


# ─────────────────────────────────────────────
#  DETECTION PRE-PROCESSING
# ─────────────────────────────────────────────

def prepare_for_detection(frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Resize and normalize frame for YOLOv8 input."""
    resized = cv2.resize(frame, (target_size, target_size))
    return resized


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """CLAHE contrast enhancement — great for dark/low-light footage."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
#  USER-FACING ENHANCEMENT FILTERS
# ─────────────────────────────────────────────

def denoise(frame: np.ndarray, strength: int = 10) -> np.ndarray:
    """Remove noise using Non-Local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


def deblur(frame: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Sharpen blurry frames using an unsharp mask."""
    blurred = cv2.GaussianBlur(frame, (0, 0), strength)
    sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    return sharpened


def sharpen(frame: np.ndarray) -> np.ndarray:
    """Apply a sharpening kernel."""
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    return cv2.filter2D(frame, -1, kernel)


def adjust_brightness_contrast(frame: np.ndarray,
                                brightness: int = 0,
                                contrast: int = 0) -> np.ndarray:
    """
    Adjust brightness (-100 to 100) and contrast (-100 to 100).
    Uses alpha/beta formula.
    """
    alpha = 1 + contrast / 100.0   # contrast factor
    beta  = brightness              # brightness shift
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def apply_filters(frame: np.ndarray, filters: dict) -> np.ndarray:
    """
    Apply a dict of enhancement filters to a single frame.

    filters keys (all optional, default off):
      - clahe       (bool)
      - denoise     (bool)
      - deblur      (bool)
      - sharpen     (bool)
      - brightness  (int, -100 to 100)
      - contrast    (int, -100 to 100)
    """
    out = frame.copy()

    if filters.get("clahe", False):
        out = apply_clahe(out)

    if filters.get("denoise", False):
        out = denoise(out)

    if filters.get("deblur", False):
        out = deblur(out)

    if filters.get("sharpen", False):
        out = sharpen(out)

    brightness = filters.get("brightness", 0)
    contrast   = filters.get("contrast",   0)
    if brightness != 0 or contrast != 0:
        out = adjust_brightness_contrast(out, brightness, contrast)

    return out


def zoom_crop(frame: np.ndarray, bbox: tuple, zoom: float = 2.0,
              out_size: tuple = (320, 240)) -> np.ndarray:
    """
    Crop and zoom into a bounding box region of the frame.
    bbox: (x1, y1, x2, y2) in pixel coords
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return cv2.resize(frame, out_size)

    crop = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(crop, out_size, interpolation=cv2.INTER_CUBIC)
    return zoomed
