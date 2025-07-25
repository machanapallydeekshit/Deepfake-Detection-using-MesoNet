"""
colour_scan.py   — revised
===========================
Flags videos whose cheek skin colour is noticeably different from the
neck/ear skin – a classic sign of face swapping.

For each .mp4 in  ./test_videos/  prints
    filename   median-ΔEab   →   COLOUR-SUSPECT / OK
--------------------------------------------------------------------
"""

import cv2
import glob
import os
import numpy as np
import mediapipe as mp

# ────────────────────────── tunables ──────────────────────────────
SAMPLE_FPS          = 5         # analyse this many frames per second
PATCH_SIZE          = 40        # initial square crop size (px)
MIN_COVER_FRAC      = 0.70      # accept patch if ≥70 % of it is inside frame
DELTAE_THRESHOLD    = 10.0      # >10 ⇒ skin-tone mismatch
MIN_FACE_HEIGHT_PX  = 80        # ignore faces smaller than this (was 100)
MIN_VALID_FRAC      = 0.40      # need ≥40 % valid frames among *sampled* ones
DARK_FRAME_L_THRESH = 30        # skip frame if too dark (CIELab L* < 30)

# Face-mesh landmark indices
NOSE_TIP   = 1
CHIN       = 152
LEFT_EAR   = 234
RIGHT_EAR  = 454

mp_mesh = mp.solutions.face_mesh

# ───────────────────────── helpers ────────────────────────────────
def safe_crop(img, centre_xy, size, min_cover_frac=MIN_COVER_FRAC):
    """
    Return a square patch centred on (x,y).
    Clamps to image borders; returns None if < min_cover_frac of the requested
    area is available.
    """
    x, y = centre_xy
    half = size // 2
    h, w = img.shape[:2]

    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)

    patch = img[y1:y2, x1:x2]
    cover = (x2 - x1) * (y2 - y1) / (size * size)

    return patch if cover >= min_cover_frac else None


def mean_ab_L(patch_bgr):
    """Return mean (a*, b*, L*) in CIELab colour-space."""
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    return float(a.mean()), float(b.mean()), float(L.mean())


# ───────────────────── per-video processing ───────────────────────
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] Cannot open {path}")
        return

    fps_orig  = cap.get(cv2.CAP_PROP_FPS) or 30
    step      = max(int(round(fps_orig / SAMPLE_FPS)), 1)
    frame_idx = 0

    deltaEs        = []
    valid_frames   = 0
    sampled_frames = 0

    with mp_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue
            sampled_frames += 1

            h, w = frame.shape[:2]
            results = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                frame_idx += 1
                continue

            lm = results.multi_face_landmarks[0].landmark
            p  = lambda idx: (int(lm[idx].x * w), int(lm[idx].y * h))

            # quick size filter
            if abs(p(CHIN)[1] - p(NOSE_TIP)[1]) < MIN_FACE_HEIGHT_PX:
                frame_idx += 1
                continue

            # ── get patches ─────────────────────────────────────────
            face_patch = safe_crop(frame, p(NOSE_TIP), PATCH_SIZE)
            neck_centre = (p(CHIN)[0], p(CHIN)[1] + PATCH_SIZE // 2)
            neck_patch = safe_crop(frame, neck_centre, PATCH_SIZE)

            if neck_patch is None:                              # fall-back to ear
                ear_idx = LEFT_EAR if lm[LEFT_EAR].visibility > lm[RIGHT_EAR].visibility else RIGHT_EAR
                neck_patch = safe_crop(frame, p(ear_idx), PATCH_SIZE)

            if face_patch is None or neck_patch is None:
                frame_idx += 1
                continue

            a_f, b_f, L_f = mean_ab_L(face_patch)
            a_n, b_n, L_n = mean_ab_L(neck_patch)

            # global illumination check
            if L_f < DARK_FRAME_L_THRESH or L_n < DARK_FRAME_L_THRESH:
                frame_idx += 1
                continue

            # ΔEab   (ignoring L because we pre-filtered by L*)
            deltaE = ((a_f - a_n) ** 2 + (b_f - b_n) ** 2) ** 0.5
            deltaEs.append(deltaE)
            valid_frames += 1
            frame_idx += 1

    cap.release()

    # ── final decision ────────────────────────────────────────────
    if not deltaEs or valid_frames / sampled_frames < MIN_VALID_FRAC:
        print(f"{os.path.basename(path):30s} : NOT ENOUGH DATA (skipped)")
        return

    median_dE = float(np.median(deltaEs))
    verdict   = "COLOUR-SUSPECT" if median_dE > DELTAE_THRESHOLD else "OK"
    print(f"{os.path.basename(path):30s}  median-ΔEab: {median_dE:5.1f}  →  {verdict}")


# ───────────────────────── batch run ──────────────────────────────
if __name__ == "__main__":
    videos = sorted(glob.glob("test_videos/*.mp4"))
    if not videos:
        print("No .mp4 files found in test_videos/")
    for vid in videos:
        process_video(vid)
