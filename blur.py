"""
blur_scan.py
============
Flags videos whose face region looks *softer / more blurred* than the
immediate surroundings – a common sign of DeepFaceLab-style warp-blur.

Dependencies
------------
pip install mediapipe opencv-python numpy

Usage
-----
python blur_scan.py
(assumes your .mp4 files live in  ./test_videos/ )
"""

import cv2
import glob
import os
import numpy as np
import mediapipe as mp

# ───────────────────────────── tunables ────────────────────────────────────────
SAMPLE_FPS           = 5           # analyse this many frames per second
BLUR_RATIO_THRESHOLD = 0.70        # < 0.70  ⇒  face softer than context ⇒ flag
MIN_FACE_HEIGHT_PX   = 100         # ignore tiny faces (low landmark accuracy)
CONTEXT_PAD_PX       = 30          # thickness of the ring around face for context
LOW_SHARPNESS_SKIP   = 30.0        # if context Laplacian var < this, skip frame
MIN_VALID_FRAC       = 0.40        # need landmarks on at least 40 % of frames

mp_fd = mp.solutions.face_detection

# ─────────────────────── helpers ───────────────────────────────────────────────
def laplacian_variance(img_gray):
    """Edge-energy score: higher = sharper."""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def context_ring(frame, bbox, pad):
    """Return a 'ring' (border band) just outside the face box."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Outer rectangle (clamped to frame)
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)

    ring = np.zeros((oy2 - oy1, ox2 - ox1), dtype=np.uint8)

    # make ring mask: 1s in outer rect minus inner face box region
    inner_x1, inner_y1 = x1 - ox1, y1 - oy1
    inner_x2, inner_y2 = inner_x1 + (x2 - x1), inner_y1 + (y2 - y1)
    ring[:, :] = 255
    ring[inner_y1:inner_y2, inner_x1:inner_x2] = 0

    # extract outer ROI & apply mask
    outer_roi = cv2.cvtColor(frame[oy1:oy2, ox1:ox2], cv2.COLOR_BGR2GRAY)
    context_pixels = outer_roi[ring == 255]
    return context_pixels

# ─────────────────── main per-video routine ───────────────────────────────────
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] Cannot open {path}")
        return None

    orig_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    step      = max(int(round(orig_fps / SAMPLE_FPS)), 1)
    frame_idx = 0

    ratios = []
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

                    face_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    face_var = laplacian_variance(face_roi_gray)

                    ctx_pixels = context_ring(frame, (x1, y1, x2, y2), CONTEXT_PAD_PX)
                    if ctx_pixels.size < 100:
                        frame_idx += 1
                        continue

                    ctx_var = np.var(cv2.Laplacian(ctx_pixels, cv2.CV_64F))
                    if ctx_var < LOW_SHARPNESS_SKIP:
                        frame_idx += 1
                        continue

                    ratio = face_var / ctx_var
                    ratios.append(ratio)
                    valid_frames += 1

            frame_idx += 1

    cap.release()

    if not ratios or valid_frames / len(ratios) < MIN_VALID_FRAC:
        print(f"{os.path.basename(path)} : NOT ENOUGH DATA (skipped)")
        return None

    median_ratio = float(np.median(ratios))
    verdict = "BLUR-SUSPECT" if median_ratio < BLUR_RATIO_THRESHOLD else "OK"

    return median_ratio


# ────────────────── run on every .mp4 in test_videos/ ─────────────────────────
if __name__ == "__main__":
    files = sorted(glob.glob("test_videos/*.mp4"))
    if not files:
        print("No .mp4 files found in test_videos/")
    for fpath in files:
        process_video(fpath)
