import cv2
import numpy as np
import os
from mtcnn import MTCNN  # MTCNN for face extraction
import mediapipe as mp
from collections import defaultdict  # <-- ADDED for region summary
from classifiers import MesoInception4
from gradcam import get_gradcam_heatmap, overlay_heatmap_on_image, compute_region_metrics

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh

def load_meso4_model(weights_path):
    model = MesoInception4()
    model.load(weights_path)
    return model

def preprocess_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def extract_faces_mtcnn(frame, detector):
    faces = detector.detect_faces(frame)
    face_images = []
    for face in faces:
        x, y, w, h = face['box']
        x1, y1, x2, y2 = max(0, x), max(0, y), x + w, y + h
        face_images.append(frame[y1:y2, x1:x2])
    return face_images

def extract_landmarks(face, face_mesh):
    results = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
    return landmarks

def annotate_heatmap_with_regions(heatmap_overlay, regions, color=(0, 255, 0), thickness=2):
    for region_name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(heatmap_overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(heatmap_overlay, region_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return heatmap_overlay

def define_regions_from_landmarks(landmarks, face_shape, overlay_shape):
    face_h, face_w = face_shape[:2]
    overlay_h, overlay_w = overlay_shape[:2]
    abs_landmarks = [(int(lx * face_w), int(ly * face_h)) for lx, ly in landmarks]
    scaled_landmarks = [
        (int(x * overlay_w / face_w), int(y * overlay_h / face_h)) for x, y in abs_landmarks
    ]
    def box_around(pt, size=5):
        cx, cy = pt
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(overlay_w - 1, x2), min(overlay_h - 1, y2)
        return (x1, y1, x2, y2)
    def box_from_points(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        left, top = max(0, left), max(0, top)
        right, bottom = min(overlay_w - 1, right), min(overlay_h - 1, bottom)
        return (left, top, right, bottom)
    left_eye = box_from_points(scaled_landmarks[33], scaled_landmarks[133])
    right_eye = box_from_points(scaled_landmarks[362], scaled_landmarks[263])
    nose = box_around(scaled_landmarks[4], size=5)
    forehead = box_around(scaled_landmarks[10], size=5)
    lips = box_from_points(scaled_landmarks[13], scaled_landmarks[14])
    regions = {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "nose": nose,
        "forehead": forehead,
        "lips": lips,
    }
    valid_regions = {}
    for region_name, (x1, y1, x2, y2) in regions.items():
        if x2 > x1 and y2 > y1:
            valid_regions[region_name] = (x1, y1, x2, y2)
    return valid_regions

def analyze_heatmap_and_generate_feedback(heatmap, regions, threshold=0.5):
    region_activations = {}
    feedback_sentences = []
    for region_name, roi in regions.items():
        metrics = compute_region_metrics(heatmap, roi, threshold=threshold)
        mean_val = metrics["mean"]
        max_val = metrics["max"]
        fraction_above = metrics["fraction_above_threshold"]
        region_activations[region_name] = {
            "mean": mean_val,
            "max": max_val,
            "fraction_above": fraction_above
        }
        if fraction_above > 0.3:
            if region_name == "lips":
                feedback_sentences.append(
                    f"The model detected significant activation in the lips region (mean={mean_val:.2f}, max={max_val:.2f})."
                )
            elif region_name in ["left_eye", "right_eye"]:
                feedback_sentences.append(
                    f"Strong activation in {region_name.replace('_',' ')} (mean={mean_val:.2f}, max={max_val:.2f})."
                )
            elif region_name == "nose":
                feedback_sentences.append(
                    f"Significant activation in the nose area (mean={mean_val:.2f}, max={max_val:.2f})."
                )
            elif region_name == "forehead":
                feedback_sentences.append(
                    f"Significant activation in the forehead (mean={mean_val:.2f}, max={max_val:.2f})."
                )
    if not feedback_sentences:
        return ("No significant activation was detected in any specific region.", region_activations)
    else:
        return (" ".join(feedback_sentences), region_activations)

def detect_deepfake(video_path, model, face_mesh, detector, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video {video_path}")
        return None

    frame_count = 0
    deepfake_scores = []
    gradcam_folder = "gradcam_outputs"
    original_folder = "original_frames"

    if not os.path.exists(gradcam_folder):
        os.makedirs(gradcam_folder)
    if not os.path.exists(original_folder):
        os.makedirs(original_folder)

    region_max_count = defaultdict(int)
    total_analysed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:
                faces = extract_faces_mtcnn(frame, detector)
                if not faces:
                    log(f"No faces detected in frame {frame_count}")
                    frame_count += 1
                    continue

                for i, face in enumerate(faces):
                    landmarks = extract_landmarks(face, face_mesh)
                    if not landmarks:
                        log(f"No landmarks detected for face {i} in frame {frame_count}")
                        continue

                    processed_face = preprocess_frame(face)
                    prediction = model.predict(processed_face)
                    deepfake_score = prediction[0][0]
                    deepfake_scores.append(deepfake_score)
                    log(f"Frame {frame_count}, Face {i}: Deepfake score = {deepfake_score:.4f}")

                    heatmap_small = get_gradcam_heatmap(
                        model=model.model,
                        img_array=processed_face,
                        last_conv_layer_name='conv_last',
                        class_index=0
                    )

                    heatmap_resized = cv2.resize(
                        heatmap_small,
                        (face.shape[1], face.shape[0])
                    )

                    heatmap_overlay = overlay_heatmap_on_image(face, heatmap_small, alpha=0.5)
                    regions = define_regions_from_landmarks(landmarks, face.shape, heatmap_resized.shape)
                    feedback, region_activations = analyze_heatmap_and_generate_feedback(heatmap_resized, regions)
                    log(f"Frame {frame_count}, Face {i} Feedback: {feedback}")

                    if region_activations:
                        max_region = None
                        max_value = -1
                        for rname, stats in region_activations.items():
                            frac_above = stats["fraction_above"]
                            if frac_above > max_value:
                                max_value = frac_above
                                max_region = rname
                        if max_region is not None:
                            region_max_count[max_region] += 1
                            total_analysed_frames += 1
                            log(f"-> The region with highest activation in this frame is {max_region} (fraction={max_value:.2f}).")

                                        # Save original face image before heatmap
                    original_outname = os.path.join(
                        original_folder,
                        f"frame_{frame_count}_face_{i}_original.jpg"
                    )
                    cv2.imwrite(original_outname, face)

                    annotated_heatmap = annotate_heatmap_with_regions(heatmap_overlay, regions)
                    outname = os.path.join(
                        gradcam_folder,
                        f"frame_{frame_count}_face_{i}_annotated.jpg"
                    )
                    cv2.imwrite(outname, annotated_heatmap)

            frame_count += 1

    finally:
        cap.release()

    if deepfake_scores:
        mean_score = np.mean(deepfake_scores)
    else:
        return None

    if total_analysed_frames > 0:
        log("=== Region Activation Summary ===")
        for region_name, count in region_max_count.items():
            log(f"{region_name} had the highest activation in {count} frames.")
        most_activated_region = max(region_max_count, key=region_max_count.get)
        log(
            f"Overall, '{most_activated_region}' was the most-activated region "
            f"in {region_max_count[most_activated_region]} out of {total_analysed_frames} analyzed frames."
        )
    else:
        log("No frames had detectable landmarks or regions to compare.")

    return mean_score
