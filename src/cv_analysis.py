"""
GeoGreen Revolution — Computer Vision Analysis Module (v3)
==========================================================
Fast, accurate land-cover classification for Google Maps/Earth
satellite screenshot images.

Approach:
  - Uses HSV + LAB color-space thresholds (not K-Means — much faster)
  - Crops Google Maps UI elements before processing
  - Applies morphological cleanup for precision
  - Draws color-coded precision contour outlines:
       GREEN  → Vegetation
       BROWN  → Barren / Fallow land
       RED    → Built-up / Urban
       BLUE   → Water bodies
"""

import cv2
import numpy as np
import os


# ── Visual style for each class ─────────────────────────────────────
CLASSES = {
    "Vegetation": {
        "mask_rgb": (34, 180, 34),
        "outline_bgr": (0, 200, 0),
        "label_bgr": (0, 255, 0),
    },
    "Water": {
        "mask_rgb": (30, 100, 255),
        "outline_bgr": (255, 100, 30),
        "label_bgr": (255, 150, 50),
    },
    "Built-up": {
        "mask_rgb": (180, 60, 60),
        "outline_bgr": (0, 0, 220),
        "label_bgr": (0, 0, 255),
    },
    "Bare/Sparse": {
        "mask_rgb": (180, 140, 80),
        "outline_bgr": (30, 105, 180),
        "label_bgr": (20, 120, 200),
    },
}


def _crop_ui(img_bgr):
    """
    Remove Google Maps UI overlays:
      - Bottom bar (scale, copyright, coords text)
      - Bottom-left minimap
      - Bottom-right controls (3D, compass, zoom)
    """
    h, w = img_bgr.shape[:2]
    bottom_crop = int(h * 0.07)
    left_crop = int(w * 0.055)
    right_crop = int(w * 0.055)

    y2 = h - bottom_crop
    x1 = left_crop
    x2 = w - right_crop

    return img_bgr[0:y2, x1:x2].copy(), (0, y2, x1, x2)


def _classify_pixels(img_bgr):
    """
    Classify each pixel using HSV + LAB thresholding.
    Returns a label image: 0=Vegetation, 1=Water, 2=Built-up, 3=Bare/Sparse
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    R, G, Bl = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)

    # Excess Green Index
    total = R + G + Bl + 1e-5
    exg = (2.0 * G - R - Bl) / total

    # Result label map: default = Bare/Sparse (3)
    labels = np.full((h, w), 3, dtype=np.uint8)

    # ── 1. VEGETATION ───────────────────────────────────────────────
    # Green in HSV: H in [25, 95], S > 25, V > 30
    # LAB: a* channel < 125 (green side of the green-red axis)
    # ExG > 0
    veg_mask = (
        (H >= 25) & (H <= 95) &
        (S >= 20) & (V >= 30) &
        (A < 125) &
        (exg > -0.05)
    )
    labels[veg_mask] = 0

    # ── 2. WATER ────────────────────────────────────────────────────
    # Blue hue: H in [85, 140], S > 20, V > 15
    # Blue-dominant pixels (Blue channel clearly exceeds Red and Green)
    water_mask = (
        ((H >= 85) & (H <= 140) & (S >= 20) & (V >= 15)) |
        # Very dark AND blue-dominant (deep/turbid water, NOT roads/shadows)
        ((V < 45) & (S < 60) & (Bl > R + 10)) |
        # Blue channel dominates significantly (e.g. clear water bodies)
        ((Bl > R + 30) & (Bl > G + 15) & (S > 25))
    )
    labels[water_mask] = 1

    # ── 3. BUILT-UP / URBAN ─────────────────────────────────────────
    # Grey tones: low saturation, moderate brightness
    # Concrete / asphalt / rooftops
    buildup_mask = (
        (S < 30) &
        (V > 60) & (V < 230) &
        (exg < 0.02) &
        (~veg_mask) & (~water_mask)
    )
    # Bluish-grey rooftops common in Indian cities — but NOT water
    # Tighten: require low saturation (< 40) so true water (S > 40) is excluded
    bluegrey_mask = (
        (H >= 90) & (H <= 130) &
        (S >= 10) & (S <= 40) &
        (V >= 80) & (V <= 200) &
        (~water_mask)   # ← critical: don't steal water pixels
    )
    labels[buildup_mask | bluegrey_mask] = 2

    # ── 4. BARE/SPARSE stays as default ─────────────────────────────
    # Warm tones (yellow/brown/tan), or anything not classified above

    return labels



CLASS_NAMES = ["Vegetation", "Water", "Built-up", "Bare/Sparse"]


def analyze_land_cover(image_path, n_clusters=5):
    """
    Segment an RGB satellite image with high accuracy.

    Parameters
    ----------
    image_path : str
        Path to the PNG/JPG image.
    n_clusters : int
        Not used in v3 (kept for API compatibility).

    Returns
    -------
    stats    : dict  – percentage coverage per class
    seg_mask : ndarray (H, W, 3) RGB – color-coded segmentation
    overlay  : ndarray (H, W, 3) RGB – original + precision outlines
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Step 1: Crop UI ─────────────────────────────────────────────
    cropped, (cy1, cy2, cx1, cx2) = _crop_ui(img_bgr)
    ch, cw = cropped.shape[:2]
    total_pixels = ch * cw

    # ── Step 2: Classify pixels ─────────────────────────────────────
    label_map = _classify_pixels(cropped)

    # ── Step 3: Morphological cleanup per class ─────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cleaned_labels = label_map.copy()

    for cls_id in range(4):
        cls_binary = (label_map == cls_id).astype(np.uint8) * 255
        cls_binary = cv2.morphologyEx(cls_binary, cv2.MORPH_CLOSE, kernel_close)
        cls_binary = cv2.morphologyEx(cls_binary, cv2.MORPH_OPEN, kernel_open)
        # Update only newly filled pixels
        newly_filled = (cls_binary > 0) & (cleaned_labels != cls_id)
        # Only fill if the pixel was previously "Bare/Sparse" (default)
        cleaned_labels[(cls_binary > 0) & (label_map == 3)] = cls_id

    label_map = cleaned_labels

    # ── Step 4: Statistics ──────────────────────────────────────────
    stats = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = np.count_nonzero(label_map == cls_id)
        stats[cls_name] = round(count / total_pixels * 100, 1)

    # ── Step 5: Build segmentation mask ─────────────────────────────
    seg_crop = np.zeros((ch, cw, 3), dtype=np.uint8)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        seg_crop[label_map == cls_id] = CLASSES[cls_name]["mask_rgb"]

    seg_full = np.full((orig_h, orig_w, 3), 50, dtype=np.uint8)
    seg_full[cy1:cy2, cx1:cx2] = seg_crop

    # ── Step 6: Precision contour outlines ──────────────────────────
    overlay = img_bgr.copy()

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        style = CLASSES[cls_name]
        cls_binary = (label_map == cls_id).astype(np.uint8) * 255

        # Extra cleanup for contour drawing
        cls_binary = cv2.morphologyEx(cls_binary, cv2.MORPH_CLOSE, kernel_close)

        if np.count_nonzero(cls_binary) == 0:
            continue

        contours, _ = cv2.findContours(
            cls_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        region_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_pixels * 0.002:
                continue

            # Offset to full-image coordinates
            cnt_shifted = cnt.copy()
            cnt_shifted[:, :, 0] += cx1
            cnt_shifted[:, :, 1] += cy1

            cv2.drawContours(overlay, [cnt_shifted], -1, style["outline_bgr"], 2)

            region_count += 1
            x, y, bw, bh = cv2.boundingRect(cnt_shifted)
            label_text = f"{cls_name} #{region_count}"

            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            ly = max(y - 5, th + 5)
            cv2.rectangle(
                overlay, (x, ly - th - 4), (x + tw + 4, ly + 2), (0, 0, 0), -1
            )
            cv2.putText(
                overlay, label_text, (x + 2, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, style["label_bgr"], 1, cv2.LINE_AA
            )

    # ── Legend ──────────────────────────────────────────────────────
    legend_y = orig_h - 35
    for cls_name in reversed(list(CLASSES.keys())):
        pct = stats.get(cls_name, 0)
        if pct > 0:
            c = CLASSES[cls_name]["outline_bgr"]
            cv2.rectangle(overlay, (orig_w - 250, legend_y - 14),
                          (orig_w - 228, legend_y), c, -1)
            cv2.putText(overlay, f"{cls_name}: {pct:.1f}%",
                        (orig_w - 224, legend_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                        1, cv2.LINE_AA)
            legend_y -= 22

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return stats, seg_full, overlay_rgb
