"""
halo_scan.py
============
Detects a “mask halo” (strong, unnatural edge around the face) that is
typical of many face-swap deepfakes.

It processes every .mp4 in  ./test_videos/  and prints:

    filename   median-halo-ratio   →   EDGE-SUSPECT / OK

A ratio > 1.50 means the inner ring (right next to the face) has a lot
more edge energy than the nearby background – a likely mask seam.

---------------------------------------------------------------
Install once:
    pip install mediapipe opencv-python numpy
Run:
    python halo_scan.py
---------------------------------------------------------------
"""

import cv2
import glob
import os
import numpy as np
import mediapipe as mp

# ───────────────────────── parameters ─────────────────────────
SAMPLE_FPS            = 5         # frames analysed per second
PAD_INNER_PX          = 10        # ring thickness right outside face
PAD_OUTER_PX          = 20        # further ring for “normal” background
HALO_RATIO_THRESHOLD  = 1.50      # >1.5 ⇒ halo likely
MIN_FACE_HEIGHT_PX    = 100       # ignore tiny faces
SHARPNESS_GLOBAL_MIN  = 30.0      # skip motion-blurred frames
MIN_VALID_FRAC        = 0.40      # need ≥40 % usable sampled frames

mp_fd = mp.solutions.face_detection

# ───────────────────── helper functions ───────────────────────
def sobel_edge_strength(gray):
    """Return mean magnitude of Sobel edges (quick edge energy)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag))

def ring_mask(h, w, bbox, inner_pad, outer_pad):
    """Binary mask for pixels between inner_pad and outer_pad around bbox."""
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), np.uint8)

    # outer rectangle
    ox1 = max(0, x1 - outer_pad)
    oy1 = max(0, y1 - outer_pad)
    ox2 = min(w, x2 + outer_pad)
    oy2 = min(h, y2 + outer_pad)
    mask[oy1:oy2, ox1:ox2] = 255

    # carve inner rectangle (up to inner_pad) as zeros
    ix1 = max(0, x1 - inner_pad)
    iy1 = max(0, y1 - inner_pad)
    ix2 = min(w, x2 + inner_pad)
    iy2 = min(h, y2 + inner_pad)
    mask[iy1:iy2, ix1:ix2] = 0
    return mask

# ───────────────────── per-video routine ──────────────────────
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] Cannot open {path}")
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step     = max(int(round(orig_fps / SAMPLE_FPS)), 1)
    frame_i  = 0

    ratios      = []
    valid_count = 0

    with mp_fd.FaceDetection(model_selection=1,
                             min_detection_confidence=0.5) as fd:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_i % step == 0:
                h, w = frame.shape[:2]

                # global sharpness check – skip very blurry frames
                if cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                 cv2.CV_64F).var() < SHARPNESS_GLOBAL_MIN:
                    frame_i += 1
                    continue

                results = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    det  = results.detections[0]
                    box  = det.location_data.relative_bounding_box
                    x1   = int(box.xmin * w)
                    y1   = int(box.ymin * h)
                    x2   = int((box.xmin + box.width) * w)
                    y2   = int((box.ymin + box.height) * h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if (y2 - y1) < MIN_FACE_HEIGHT_PX:
                        frame_i += 1
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # inner halo (ring between PAD_INNER and face edge)
                    inner_mask = ring_mask(h, w, (x1, y1, x2, y2),
                                           0, PAD_INNER_PX)
                    inner_strength = sobel_edge_strength(
                        cv2.bitwise_and(gray, gray, mask=inner_mask))

                    # outer ring (PAD_OUTER to PAD_OUTER+PAD_INNER)
                    outer_mask = ring_mask(h, w, (x1, y1, x2, y2),
                                           PAD_INNER_PX, PAD_OUTER_PX)
                    outer_strength = sobel_edge_strength(
                        cv2.bitwise_and(gray, gray, mask=outer_mask))

                    if outer_strength < 1e-3:        # avoid divide by zero
                        frame_i += 1
                        continue

                    ratio = inner_strength / outer_strength
                    ratios.append(ratio)
                    valid_count += 1

            frame_i += 1

    cap.release()

    if not ratios or valid_count / len(ratios) < MIN_VALID_FRAC:
        print(f"{os.path.basename(path)} : NOT ENOUGH DATA (skipped)")
        return None

    median_ratio = float(np.median(ratios))
    verdict = "EDGE-SUSPECT" if median_ratio > HALO_RATIO_THRESHOLD else "OK"

    print(f"{os.path.basename(path):30s}  median-halo-ratio: {median_ratio:0.2f}  →  {verdict}")

    return median_ratio

# ─────────────────────── run all .mp4s ────────────────────────
if __name__ == "__main__":
    vids = sorted(glob.glob("test_videos/*.mp4"))
    if not vids:
        print("No .mp4 files found in test_videos/")
    for vid in vids:
        process_video(vid)
