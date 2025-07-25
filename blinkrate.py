"""
quick_blink_scan.py
-------------------
Counts eye-blinks in every video inside the local folder  `test_videos/`
and prints the blink-per-minute rate plus a “SUSPICIOUS” flag
if the rate is lower than a chosen threshold.

Dependencies
------------
pip install mediapipe opencv-python numpy

Usage
-----
python quick_blink_scan.py
"""

import cv2
import glob
import os
from collections import deque
from math import hypot
import numpy as np
import mediapipe as mp

# ────────────────────────────────────────────────────────────────────────────────
# 1. Tunable constants
# ────────────────────────────────────────────────────────────────────────────────
SAMPLE_FPS           = 5                     # how many frames per second to sample
EAR_THRESH_DEFAULT   = 0.21                  # EAR below this → eye considered shut
CONSEC_FRAMES_BLINK  = 1                     # need this many shut frames in a row
BLINK_RATE_WARNING   = 6                     # < 6 blinks-per-minute → flag
MIN_VALID_FRAC       = 0.50                  # need landmarks on ≥50 % of sampled frames
MIN_FACE_HEIGHT_PX   = 100                  # ignore faces smaller than this

# MediaPipe landmark indices for left & right eye
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh

# ────────────────────────────────────────────────────────────────────────────────
# 2. Helper functions
# ────────────────────────────────────────────────────────────────────────────────
def _euclidean(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(landmarks, eye_indices):
    """Compute EAR from six eye landmark points (x, y in pixels)."""
    # vertical pairs (p2-p6, p3-p5) and horizontal pair (p1-p4)
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vert1 = _euclidean(p2, p6)
    vert2 = _euclidean(p3, p5)
    horiz = _euclidean(p1, p4)
    # avoid division by zero
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)

def blink_rate_from_ear_sequence(ears, fps):
    """Return total blink count and blinks-per-minute from EAR time-series."""
    blink_count = 0
    shut_frames = 0
    threshold   = EAR_THRESH_DEFAULT

    # Optional: adaptive threshold (median of first 2 s × 0.75)
    warmup = int(min(len(ears), fps * 2))
    if warmup:  # median EAR of first 2 s
        threshold = 0.75 * float(np.median(ears[:warmup]))

    for ear in ears:
        if ear < threshold:
            shut_frames += 1
        else:
            if shut_frames >= CONSEC_FRAMES_BLINK:
                blink_count += 1
            shut_frames = 0
    # handle end-of-sequence blink
    if shut_frames >= CONSEC_FRAMES_BLINK:
        blink_count += 1

    minutes = len(ears) / fps / 60.0
    rate    = blink_count / minutes if minutes else 0.0
    return blink_count, rate

# ────────────────────────────────────────────────────────────────────────────────
# 3. Main video loop
# ────────────────────────────────────────────────────────────────────────────────
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] Could not open {path}")
        return None  # return explicitly

    orig_fps  = cap.get(cv2.CAP_PROP_FPS) or 30   # default to 30 if unknown
    step      = max(int(round(orig_fps / SAMPLE_FPS)), 1)
    frame_idx = 0

    ears = []               # EAR time-series
    valid_frames = 0

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                h, w = frame.shape[:2]
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark

                    # convert normalised → pixel coords
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                    # discard if face too small
                    face_height = max([p[1] for p in pts]) - min([p[1] for p in pts])
                    if face_height < MIN_FACE_HEIGHT_PX:
                        ears.append(0.5)   # push dummy open-eye EAR
                        frame_idx += 1
                        continue

                    left_ear  = eye_aspect_ratio(pts, LEFT_EYE)
                    right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
                    ears.append((left_ear + right_ear) / 2.0)
                    valid_frames += 1
                else:
                    ears.append(0.5)

            frame_idx += 1

    cap.release()

    processed = len(ears)
    if processed == 0 or valid_frames / processed < MIN_VALID_FRAC:
        print(f"{os.path.basename(path)} : NOT ENOUGH LANDMARK DATA (skipped)")
        return None

    blink_count, bpm = blink_rate_from_ear_sequence(ears, SAMPLE_FPS)
    flag = "SUSPICIOUS" if bpm < BLINK_RATE_WARNING else "OK"

    return {
        "blink_count": blink_count,
        "bpmrate": bpm
    }


# ────────────────────────────────────────────────────────────────────────────────
# 4. Run on every file in test_videos/
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    files = sorted(glob.glob("test_videos/*.mp4"))  # adapt pattern if needed
    if not files:
        print("No .mp4 files found in test_videos/")
    for fp in files:
        process_video(fp)
