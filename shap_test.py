import cv2
import numpy as np
import os
try:
    from mtcnn.mtcnn import MTCNN
except ImportError:
    raise ModuleNotFoundError("The 'mtcnn' module is not installed. Install it using 'pip install mtcnn'.")
from classifiers import MesoInception4  # Import Meso4 from classifiers.py
import shap  # Import SHAP
import matplotlib.pyplot as plt  # For visualizing SHAP outputs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.python.framework import ops
import tensorflow as tf

# Register gradient for LeakyReLU
@ops.RegisterGradient("shap_LeakyRelu")
def _leaky_relu_grad(op, grad):
    return tf.where(op.inputs[0] >= 0, grad, grad * 0.01)  # Default slope for LeakyReLU is 0.01

def log(message):
    print(f"[DEBUG] {message}")

# Load the Meso4 model
def load_meso4_model(weights_path):
    log("Initializing Meso4 model.")
    model = MesoInception4()  # Instantiate the Meso4 class
    log("Loading weights.")
    model.load(weights_path)  # Load the pretrained weights
    return model.model  # Access the internal Keras model

# Preprocess a single frame for prediction
def preprocess_frame(frame):
    log("Preprocessing frame.")
    # Resize frame to 256x256 as expected by the Meso4 model
    frame = cv2.resize(frame, (256, 256))
    # Normalize pixel values to [0, 1]
    frame = frame / 255.0
    # Add batch dimension
    return np.expand_dims(frame, axis=0)

# Detect and extract faces using MTCNN
def extract_faces_mtcnn(frame, detector):
    log("Detecting faces.")
    faces = detector.detect_faces(frame)
    face_images = []
    for face in faces:
        box = face['box']
        x, y, w, h = box
        x1, y1, x2, y2 = max(0, x), max(0, y), x + w, y + h
        face_images.append(frame[y1:y2, x1:x2])
    log(f"Detected {len(face_images)} face(s).")
    return face_images

# Visualize SHAP explanations
def visualize_shap(explainer, input_data, frame_count, face_index):
    log("Generating SHAP explanations.")
    shap_values = explainer.shap_values(input_data, check_additivity=False)

    # Normalize SHAP values for visualization
    shap_values_normalized = [
        (value - np.min(value)) / (np.max(value) - np.min(value) + 1e-8)
        for value in shap_values
    ]

    # SHAP visualization
    shap.image_plot(shap_values_normalized, input_data, show=False)

    # Ensure the output directory exists
    output_dir = "shap_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the SHAP visualization as an image
    output_path = os.path.join(output_dir, f"frame_{frame_count}_face_{face_index}.png")
    plt.savefig(output_path)
    log(f"SHAP visualization saved to {output_path}.")
    plt.close()


# Test the model on a video file
def detect_deepfake(video_path, model, detector, explainer):
    log(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video {video_path}")
        return None

    frame_count = 0
    deepfake_scores = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 10th frame for efficiency
            if frame_count % 10 == 0:
                log(f"Processing frame {frame_count}.")
                faces = extract_faces_mtcnn(frame, detector)  # Extract faces
                if not faces:
                    log(f"No faces detected in frame {frame_count}.")
                    continue

                for i, face in enumerate(faces):
                    processed_face = preprocess_frame(face)
                    # Get prediction: model outputs a score, where higher score indicates deepfake
                    prediction = model.predict(processed_face)
                    deepfake_scores.append(prediction[0][0])
                    log(f"Frame {frame_count}, Face {i}: Deepfake score = {prediction[0][0]:.4f}")

                    # Generate SHAP explanations for the face
                    visualize_shap(explainer, processed_face, frame_count, i)

            frame_count += 1

    finally:
        cap.release()

    if deepfake_scores:
        avg_score = np.mean(deepfake_scores)
        log(f"Average deepfake score: {avg_score:.4f}")
        return avg_score
    else:
        log("No deepfake scores calculated.")
        return None

# Main testing function
def main():
    weights_path = "weights/MesoInception_DF.h5"  # Path to MesoNet weights
    test_video_dir = "test_videos"  # Directory containing test videos

    log("Loading MTCNN face detector.")
    detector = MTCNN()

    log("Loading Meso4 model.")
    meso4_model = load_meso4_model(weights_path)

    log("Initializing SHAP explainer.")
    explainer = shap.DeepExplainer(meso4_model, np.random.random((1, 256, 256, 3)))

    for video_file in os.listdir(test_video_dir):
        video_path = os.path.join(test_video_dir, video_file)
        if video_path.endswith((".mp4", ".avi", ".mov")):  # Add more formats if needed
            log(f"Starting analysis for video: {video_file}")
            score = detect_deepfake(video_path, meso4_model, detector, explainer)
            if score is not None:
                log(f"Final deepfake score for {video_file}: {score:.4f}")
            else:
                log(f"No faces detected or error processing video: {video_file}")

if __name__ == "__main__":
    main()
