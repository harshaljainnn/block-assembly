"""
Block Cropper Module: Extracts focused, normalized assembly ROIs.
Anchors on the central block assembly (Green beam backbone) and crops a centered,
padded region to discard desk clutter, background shadows, and hand occlusion.
"""

import cv2
import numpy as np


def crop_assembly(img, pad_ratio=0.22, min_size=180, ignore_hud=True):
    """
    Locates the active block assembly and returns a tight, square cropped image and bounding box.
    
    Args:
        img: Input BGR image (numpy array)
        pad_ratio: Padding fraction added around the assembly bounding box (default: 0.22)
        min_size: Minimum crop width/height in pixels
        ignore_hud: If True, excludes top banner (y < 160) and bottom bar (y > h - 35)
        
    Returns:
        (crop_img, (x, y, w, h)) if assembly found, else (None, (0, 0, img_w, img_h))
    """
    if img is None or img.size == 0:
        return None, (0, 0, 0, 0)

    h, w = img.shape[:2]
    mask_roi = np.ones((h, w), dtype=bool)
    if ignore_hud and h > 200:
        mask_roi[:160, :] = False
        mask_roi[h - 35:, :] = False

    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    # Plastic color masks (strict thresholds rejecting human skin tone)
    blue = (b > 70) & (b > 1.15 * r.astype(np.float32)) & (hue >= 90) & (hue <= 135) & (sat > 40) & mask_roi
    green = (g > 60) & (g > 1.10 * r.astype(np.float32)) & (hue >= 35) & (hue <= 85) & (sat > 40) & mask_roi
    red = (r > 100) & (r > 1.30 * g.astype(np.float32)) & ((hue <= 14) | (hue >= 166)) & (sat > 80) & mask_roi
    yellow = (r > 90) & (g > 75) & (r > 1.30 * b.astype(np.float32)) & (hue >= 15) & (hue <= 35) & (sat > 70) & mask_roi

    combined = (blue | green | red | yellow).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > 350]
    if not valid_cnts:
        return None, (0, 0, w, h)

    # Find the main assembly cluster
    # Anchor on the green beam if present (the spine of Steps 1-8)
    green_mask = green.astype(np.uint8) * 255
    g_cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    g_valid = [c for c in g_cnts if cv2.contourArea(c) > 500]

    if g_valid:
        g_valid.sort(key=cv2.contourArea, reverse=True)
        gx, gy, gw, gh = cv2.boundingRect(g_valid[0])
        gcx, gcy = gx + gw // 2, gy + gh // 2
        # Gather all block contours physically near the green beam
        assembly_cnts = []
        for c in valid_cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            bcx, bcy = bx + bw // 2, by + bh // 2
            if np.hypot(gcx - bcx, gcy - bcy) < max(w, h) * 0.32:
                assembly_cnts.append(c)
    else:
        valid_cnts.sort(key=cv2.contourArea, reverse=True)
        assembly_cnts = [valid_cnts[0]]

    if not assembly_cnts:
        return None, (0, 0, w, h)

    all_pts = np.vstack([c.reshape(-1, 2) for c in assembly_cnts])
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    bw, bh = max_x - min_x, max_y - min_y

    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2
    side = int(max(bw, bh) * (1.0 + pad_ratio * 2))
    side = max(side, min_size)

    # Compute square ROI centered on assembly
    x1 = int(max(0, cx - side // 2))
    y1 = int(max(0, cy - side // 2))
    x2 = int(min(w, x1 + side))
    y2 = int(min(h, y1 + side))

    # Re-adjust to maintain square shape if near frame boundaries
    side_actual = max(x2 - x1, y2 - y1)
    x1 = int(max(0, cx - side_actual // 2))
    y1 = int(max(0, cy - side_actual // 2))
    x2 = int(min(w, x1 + side_actual))
    y2 = int(min(h, y1 + side_actual))

    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2 - x1, y2 - y1)


def is_workspace_empty(img, min_block_area=400):
    """
    Fast binary check to determine if the workspace has zero blocks (Step 0).
    """
    crop, _ = crop_assembly(img, min_size=100)
    return crop is None
