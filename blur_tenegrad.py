"""
blur_scan.py
============
Flags videos whose face region looks *softer / more blurred* than the
immediate surroundings – a common sign of DeepFaceLab-style warp-blur,
now using Tenengrad focus measure.

python blur_scan.py        (expects .mp4 files inside ./test_videos/)
"""

import cv2
import glob
import os
import numpy as np
import mediapipe as mp

# ───────────────────────────── tunables ────────────────────────────────────────
SAMPLE_FPS           = 5
BLUR_RATIO_THRESHOLD = 0.40
MIN_FACE_HEIGHT_PX   = 100
CONTEXT_PAD_PX       = 10
LOW_SHARPNESS_SKIP   = 80.0
MIN_VALID_FRAC       = 0.40

mp_fd = mp.solutions.face_detection

# ─────────────────────── helpers ───────────────────────────────────────────────
def tenengrad_map(img_gray):
    """Return gradient-magnitude image (Sobel)."""
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)        # √(Gx² + Gy²)

def tenengrad_variance(img_gray):
    """Single-number focus score (variance of gradient magnitude)."""
    return tenengrad_map(img_gray).var()

def context_ring_grad(frame, bbox, pad):
    """Return gradient-magnitudes of the ring just outside the face box."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)

    outer_roi = cv2.cvtColor(frame[oy1:oy2, ox1:ox2], cv2.COLOR_BGR2GRAY)
    gmag      = tenengrad_map(outer_roi)                                     # ★

    # build ring mask (1 = context pixels)
    ring_mask = np.ones_like(gmag, dtype=bool)
    inner_x1, inner_y1 = x1 - ox1, y1 - oy1
    inner_x2, inner_y2 = inner_x1 + (x2 - x1), inner_y1 + (y2 - y1)
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    return gmag[ring_mask]                                                  # ★

# ─────────────────── main per-video routine ───────────────────────────────────
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] Cannot open {path}")
        return

    orig_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    step      = max(int(round(orig_fps / SAMPLE_FPS)), 1)
    frame_idx = 0

    ratios       = []
    valid_frames = 0

    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                h, w = frame.shape[:2]
                results = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.detections:
                    det = results.detections[0]
                    box = det.location_data.relative_bounding_box
                    x1 = int(box.xmin * w)
                    y1 = int(box.ymin * h)
                    x2 = int((box.xmin + box.width) * w)
                    y2 = int((box.ymin + box.height) * h)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if (y2 - y1) < MIN_FACE_HEIGHT_PX:
                        frame_idx += 1
                        continue

                    face_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    face_var  = tenengrad_variance(face_gray)               # ★

                    ctx_vals  = context_ring_grad(frame, (x1, y1, x2, y2), CONTEXT_PAD_PX)
                    if ctx_vals.size < 100:
                        frame_idx += 1
                        continue
                    ctx_var = ctx_vals.var()                                # ★

                    if ctx_var < LOW_SHARPNESS_SKIP:
                        frame_idx += 1
                        continue

                    ratios.append(face_var / ctx_var)
                    valid_frames += 1

            frame_idx += 1

    cap.release()

    if not ratios or valid_frames / len(ratios) < MIN_VALID_FRAC:
        print(f"{os.path.basename(path)} : NOT ENOUGH DATA (skipped)")
        return

    median_ratio = float(np.median(ratios))
    verdict = "BLUR-SUSPECT" if median_ratio < BLUR_RATIO_THRESHOLD else "OK"

    print(f"{os.path.basename(path):30s}  median-ratio: {median_ratio:0.2f}  →  {verdict}")

# ────────────────── run on every .mp4 in test_videos/ ─────────────────────────
if __name__ == "__main__":
    files = sorted(glob.glob("test_videos/*.mp4"))
    if not files:
        print("No .mp4 files found in test_videos/")
    for fpath in files:
        process_video(fpath)
