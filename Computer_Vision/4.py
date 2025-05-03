import cv2
import mediapipe as mp
import numpy as np

def detect_pose(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results.pose_landmarks

def overlay_outfit(person_img, outfit_img):
    landmarks = detect_pose(person_img)
    if not landmarks:
        print("No pose detected!")
        return person_img
    
    h, w, _ = person_img.shape
    left_shoulder = (int(landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x * w),
                     int(landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y * h))
    right_shoulder = (int(landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                      int(landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    left_hip = (int(landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].x * w),
                int(landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].y * h))
    right_hip = (int(landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x * w),
                 int(landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y * h))
    
    outfit_width = abs(right_shoulder[0] - left_shoulder[0]) * 1.5
    outfit_height = abs(left_hip[1] - left_shoulder[1]) * 1.2
    
    x_offset = max(0, left_shoulder[0] - int(outfit_width * 0.25))
    y_offset = max(0, left_shoulder[1])
    
    outfit_resized = cv2.resize(outfit_img, (int(outfit_width), int(outfit_height)))
    
    if y_offset + outfit_resized.shape[0] > h or x_offset + outfit_resized.shape[1] > w:
        print("Warning: Outfit exceeds image bounds. Adjusting...")
        outfit_resized = outfit_resized[:h - y_offset, :w - x_offset]
    
    if outfit_resized.shape[-1] == 4:  # Check if the image has an alpha channel
        alpha_channel = outfit_resized[:, :, 3] / 255.0
        alpha_channel = np.expand_dims(alpha_channel, axis=-1)  # Ensure shape compatibility
        blended_region = (alpha_channel * outfit_resized[:, :, :3] +
                          (1 - alpha_channel) * person_img[y_offset:y_offset + outfit_resized.shape[0],
                                                            x_offset:x_offset + outfit_resized.shape[1]]).astype(np.uint8)
        person_img[y_offset:y_offset + outfit_resized.shape[0], x_offset:x_offset + outfit_resized.shape[1]] = blended_region
    else:
        person_img[y_offset:y_offset + outfit_resized.shape[0], x_offset:x_offset + outfit_resized.shape[1]] = outfit_resized
    
    return person_img

if __name__ == "__main__":
    person_img = cv2.imread("person.jpg")  
    outfit_img = cv2.imread("outfit.png", cv2.IMREAD_UNCHANGED)  
    
    if person_img is None:
        print("Error: person.jpg not found or cannot be read.")
    if outfit_img is None:
        print("Error: outfit.png not found or cannot be read.")
    
    if person_img is not None and outfit_img is not None:
        output = overlay_outfit(person_img, outfit_img)
        cv2.imshow("Virtual Outfit", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
