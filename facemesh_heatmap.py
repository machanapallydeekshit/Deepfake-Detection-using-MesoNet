import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Specify input image folder containing heatmaps
input_image_folder = "heatmap_images"

# Get the first heatmap image from the folder
image_files = [f for f in os.listdir(input_image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    print("No image files found in the directory. Exiting.")
    exit()

input_image_path = os.path.join(input_image_folder, image_files[0])

# Read the heatmap image
heatmap_image = cv2.imread(input_image_path)
original_frame = heatmap_image.copy()

# Convert the BGR image to RGB
rgb_frame = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)

# Process the image to detect face landmarks
results = face_mesh.process(rgb_frame)

# Analyze the heatmap for high-activation regions
gray_heatmap = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2GRAY)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_heatmap)

# Print high-activation region
print(f"High activation region: {max_loc} with intensity {max_val}")

# Draw high-activation region on the image
cv2.circle(original_frame, max_loc, 10, (0, 0, 255), 2)

# Detect and print landmarks if available
if results.multi_face_landmarks:
    h, w, _ = heatmap_image.shape  # Image dimensions
    for face_landmarks in results.multi_face_landmarks:
        # Extract and analyze regions around key landmarks
        def extract_region(landmark, radius=10):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            x1, y1 = max(cx - radius, 0), max(cy - radius, 0)
            x2, y2 = min(cx + radius, w), min(cy + radius, h)
            return gray_heatmap[y1:y2, x1:x2]

        regions = {
            "Left Eye": extract_region(face_landmarks.landmark[159]),
            "Right Eye": extract_region(face_landmarks.landmark[386]),
            "Lips": extract_region(face_landmarks.landmark[0]),
            "Forehead": extract_region(face_landmarks.landmark[10])
        }

        # Calculate the mean intensity for each region
        for region_name, region in regions.items():
            mean_intensity = np.mean(region)
            print(f"{region_name} mean intensity: {mean_intensity}")

        # Visualize the regions
        for name, landmark in zip(["Left Eye", "Right Eye", "Lips", "Forehead"], [159, 386, 0, 10]):
            cx, cy = int(face_landmarks.landmark[landmark].x * w), int(face_landmarks.landmark[landmark].y * h)
            cv2.rectangle(original_frame, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 255, 255), 1)

# Display the annotated image
cv2.imshow('Heatmap Analysis', original_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
