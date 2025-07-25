import cv2
import mediapipe as mp
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Drawing utilities for visualization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Specify input video folder
input_video_folder = "test_videos"

# Get the first video file from the folder
video_files = [f for f in os.listdir(input_video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
if not video_files:
    print("No video files found in the directory. Exiting.")
    exit()

input_video_path = os.path.join(input_video_folder, video_files[0])

# Start video capture
cap = cv2.VideoCapture(input_video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    frame_count += 1

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect face landmarks
    results = face_mesh.process(rgb_frame)

    # Process every 10th frame
    if frame_count % 10 == 0 and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape  # Frame dimensions

            # Extract key landmark coordinates
            left_eye = face_landmarks.landmark[159]  # Example index for left eye
            right_eye = face_landmarks.landmark[386]  # Example index for right eye
            lips = face_landmarks.landmark[0]  # Example index for lips center
            forehead = face_landmarks.landmark[10]  # Example index for forehead

            # Convert normalized coordinates to pixel values
            def landmark_to_pixel(landmark):
                return int(landmark.x * w), int(landmark.y * h), landmark.z

            left_eye_coords = landmark_to_pixel(left_eye)
            right_eye_coords = landmark_to_pixel(right_eye)
            lips_coords = landmark_to_pixel(lips)
            forehead_coords = landmark_to_pixel(forehead)

            # Print coordinates
            print(f"Left Eye: {left_eye_coords}")
            print(f"Right Eye: {right_eye_coords}")
            print(f"Lips: {lips_coords}")
            print(f"Forehead: {forehead_coords}")

            # Optionally, visualize these points on the frame
            cv2.circle(frame, (left_eye_coords[0], left_eye_coords[1]), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_eye_coords[0], right_eye_coords[1]), 5, (0, 255, 0), -1)
            cv2.circle(frame, (lips_coords[0], lips_coords[1]), 5, (255, 0, 0), -1)
            cv2.circle(frame, (forehead_coords[0], forehead_coords[1]), 5, (255, 255, 0), -1)

    # Display the annotated frame
    cv2.imshow('MediaPipe FaceMesh', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
