"""
colour_scan.py
==============
Flags videos whose cheek-skin colour is noticeably different from the
neck/ear skin – a classic hint the face area was pasted in.

For every .mp4 inside  ./test_videos/  the script prints

    filename   median-ΔEab   →   COLOUR-SUSPECT / OK

--------------------------------------------------------------------
Install once:
    pip install mediapipe opencv-python numpy
Run:
    python colour_scan.py
--------------------------------------------------------------------
"""

import cv2
import glob
import os
import numpy as np
import mediapipe as mp

# ─────────────────────────── tunables ─────────────────────────────
SAMPLE_FPS          = 5        # frames per second to test
PATCH_SIZE          = 40       # square crop size (px)
DELTAE_THRESHOLD    = 12.0     # >12 ⇒ skin-tone mismatch
MIN_FACE_HEIGHT_PX  = 50      # ignore tiny faces (landmarks unreliable)
MIN_VALID_FRAC      = 0.40     # need ≥40 % valid frames
DARK_FRAME_L_THRESH = 30       # skip frame if too dark (L* < 30)

# MediaPipe landmark indices
NOSE_TIP = 1          # centre of face
CHIN     = 152        # lowest chin point
LEFT_EAR = 234        # mid ear helix
RIGHT_EAR = 454

mp_mesh = mp.solutions.face_mesh

# ───────────────────────── helpers ────────────────────────────────
def crop_square(img, center_xy, size):
    """Return size×size patch centred on (x,y); None if out of bounds."""
    x, y = center_xy
    half = size // 2
    h, w = img.shape[:2]
    x1, y1 = x - half, y - half
    x2, y2 = x + half, y + half
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        return None
    return img[y1:y2, x1:x2]

def mean_ab(patch_bgr):
    """Return mean A and B of a patch converted to CIELAB."""
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    # split -> ignore L channel
    _, a, b = cv2.split(lab)
    return float(a.mean()), float(b.mean()), float(lab[:, :, 0].mean())

# ───────────────────── per-video processing ───────────────────────
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] Cannot open {path}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step     = max(int(round(orig_fps / SAMPLE_FPS)), 1)
    frame_i  = 0

    deltaEs   = []
    valid_cnt = 0

    with mp_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % step != 0:
                frame_i += 1
                continue

            h, w = frame.shape[:2]
            results = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                frame_i += 1
                continue

            lm = results.multi_face_landmarks[0].landmark

            # Convert all landmark coords to pixel int
            px = lambda idx: (int(lm[idx].x * w), int(lm[idx].y * h))

            # Ignore if face too small
            face_height = abs(px(CHIN)[1] - px(NOSE_TIP)[1])
            if face_height < MIN_FACE_HEIGHT_PX:
                print("face height is less")
                frame_i += 1
                continue

            # Grab patches
            face_patch = crop_square(frame, px(NOSE_TIP), PATCH_SIZE)
            neck_center = (px(CHIN)[0], px(CHIN)[1] + PATCH_SIZE // 2)
            neck_patch = crop_square(frame, neck_center, PATCH_SIZE)

            # fallback to ear if neck hidden
            if neck_patch is None:
                ear_idx = LEFT_EAR if lm[LEFT_EAR].visibility > lm[RIGHT_EAR].visibility else RIGHT_EAR
                neck_patch = crop_square(frame, px(ear_idx), PATCH_SIZE)

            if face_patch is None or neck_patch is None:
                frame_i += 1
                continue

            a_face, b_face, l_face = mean_ab(face_patch)
            a_neck, b_neck, l_neck = mean_ab(neck_patch)

            # skip very dark frames
            if l_face < DARK_FRAME_L_THRESH or l_neck < DARK_FRAME_L_THRESH:
                frame_i += 1
                continue

            deltaE = ((a_face - a_neck)**2 + (b_face - b_neck)**2) ** 0.5
            deltaEs.append(deltaE)
            valid_cnt += 1
            frame_i += 1

    cap.release()

    if not deltaEs or valid_cnt / len(deltaEs) < MIN_VALID_FRAC:
        print(deltaEs,valid_cnt)
        print(f"{os.path.basename(path)} : NOT ENOUGH DATA (skipped)")
        return

    median_dE = float(np.median(deltaEs))
    verdict   = "COLOUR-SUSPECT" if median_dE > DELTAE_THRESHOLD else "OK"

    print(f"{os.path.basename(path):30s}  median-ΔEab: {median_dE:5.1f}  →  {verdict}")

    return median_dE

# ───────────────────── run every video ───────────────────────────
if __name__ == "__main__":
    vids = sorted(glob.glob("test_videos/*.mp4"))
    if not vids:
        print("No .mp4 files found in test_videos/")
    for v in vids:
        process_video(v)
