import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Load outfit image (ensure it has transparency)
outfit = cv2.imread("outfit.png", cv2.IMREAD_UNCHANGED)

# Function to overlay outfit
def overlay_outfit(frame, outfit, landmarks):
    h, w, _ = frame.shape
    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
    right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

    # Calculate outfit position and size
    outfit_width = int(np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)))
    outfit_height = int(np.linalg.norm(np.array(left_shoulder) - np.array(left_hip)))

    # Resize outfit
    outfit_resized = cv2.resize(outfit, (outfit_width, outfit_height))

    # Ensure the outfit is within frame bounds
    y_start = max(0, left_shoulder[1])
    y_end = min(h, y_start + outfit_resized.shape[0])
    x_start = max(0, left_shoulder[0])
    x_end = min(w, x_start + outfit_resized.shape[1])

    # Overlay outfit on frame
    for i in range(y_end - y_start):
        for j in range(x_end - x_start):
            if outfit_resized[i, j, 3] != 0:  # Check alpha channel
                frame[y_start + i, x_start + j] = outfit_resized[i, j, :3]

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Overlay outfit
        frame = overlay_outfit(frame, outfit, results.pose_landmarks.landmark)

    # Display frame
    cv2.imshow("Virtual Fitting Room", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()