import cv2
import numpy as np
import os
import mediapipe as mp

class VirtualFittingRoom:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_body_landmarks(self, image):
        """Detect body landmarks using MediaPipe Pose."""
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and get pose results
        results = self.pose.process(image_rgb)
        return results

    def estimate_torso_region(self, image, landmarks):
        """Estimate the torso region based on pose landmarks."""
        h, w = image.shape[:2]
        
        # Get relevant landmarks for torso
        # Using shoulders, hips and mid-points for torso estimation
        if not landmarks.pose_landmarks:
            return None
        
        lm = landmarks.pose_landmarks.landmark
        
        # Get key points for torso
        left_shoulder = (int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), 
                         int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w), 
                          int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        left_hip = (int(lm[self.mp_pose.PoseLandmark.LEFT_HIP].x * w), 
                    int(lm[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
        right_hip = (int(lm[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w), 
                     int(lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        
        # Calculate approximate torso width and height
        shoulder_width = np.sqrt((right_shoulder[0] - left_shoulder[0])**2 + 
                               (right_shoulder[1] - left_shoulder[1])**2)
        torso_height = np.sqrt((left_shoulder[0] - left_hip[0])**2 + 
                              (left_shoulder[1] - left_hip[1])**2)
        
        # Calculate torso center
        torso_center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) // 4
        torso_center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) // 4
        
        # Add some padding
        padding_width = int(shoulder_width * 0.2)
        padding_height = int(torso_height * 0.1)
        
        # Calculate torso region
        torso_region = {
            'center': (torso_center_x, torso_center_y),
            'width': int(shoulder_width + padding_width),
            'height': int(torso_height + padding_height),
            'shoulder_points': (left_shoulder, right_shoulder),
            'hip_points': (left_hip, right_hip),
            'rotation': np.arctan2(right_shoulder[1] - left_shoulder[1], 
                                  right_shoulder[0] - left_shoulder[0]) * 180 / np.pi
        }
        
        return torso_region

    def overlay_outfit(self, human_image_path, outfit_image_path, output_path):
        """
        Enhanced method to overlay an outfit onto a human image with proper alignment.
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load images
        try:
            # Load human image
            human_img = cv2.imread(human_image_path)
            if human_img is None:
                print(f"ERROR: Could not load human image from {human_image_path}")
                return False
            
            # Load outfit with transparency
            outfit_img = cv2.imread(outfit_image_path, cv2.IMREAD_UNCHANGED)
            if outfit_img is None:
                print(f"ERROR: Could not load outfit image from {outfit_image_path}")
                return False
            
            # Check if outfit has alpha channel
            if outfit_img.shape[2] != 4:
                print(f"ERROR: Outfit image doesn't have transparency (alpha channel)")
                return False
            
            print(f"Human image shape: {human_img.shape}")
            print(f"Outfit image shape: {outfit_img.shape}")
        except Exception as e:
            print(f"ERROR loading images: {e}")
            return False
        
        # Get dimensions
        h_height, h_width = human_img.shape[:2]
        o_height, o_width = outfit_img.shape[:2]
        
        # Detect body pose
        landmarks = self.detect_body_landmarks(human_img)
        
        # Get torso region for accurate placement
        torso_region = self.estimate_torso_region(human_img, landmarks)
        
        # Debug visualization of landmarks
        debug_img = human_img.copy()
        if landmarks.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                debug_img, 
                landmarks.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            # Save debug image
            debug_path = output_path.replace('.jpg', '_debug.jpg')
            cv2.imwrite(debug_path, debug_img)
            print(f"Debug image with landmarks saved to {debug_path}")
        
        if torso_region is None:
            print("WARNING: Could not estimate torso region. Falling back to center placement.")
            # Fall back to center placement
            x_offset = (h_width - o_width) // 2
            y_offset = (h_height - o_height) // 2
            scale = min(1.0, (h_width * 0.6) / o_width, (h_height * 0.6) / o_height)
            rotation_angle = 0
            center_x, center_y = h_width // 2, h_height // 2
        else:
            # Calculate scaling based on torso proportions
            torso_width = torso_region['width']
            torso_height = torso_region['height']
            
            # Scale outfit to fit the torso
            scale_x = torso_width / o_width
            scale_y = torso_height / o_height
            scale = min(scale_x, scale_y) * 0.95  # Slightly smaller to avoid overflow
            
            # Calculate the center position based on torso
            center_x, center_y = torso_region['center']
            
            # Get rotation angle from shoulder slope
            rotation_angle = torso_region['rotation']
            
            print(f"Torso width: {torso_width}, height: {torso_height}")
            print(f"Using scale factor: {scale}, rotation: {rotation_angle}°")
        
        # Resize outfit according to scale
        new_width = int(o_width * scale)
        new_height = int(o_height * scale)
        outfit_img = cv2.resize(outfit_img, (new_width, new_height))
        
        # Adjust position for outfit to center on torso
        if torso_region is None:
            # Center of image
            x_offset = (h_width - new_width) // 2
            y_offset = (h_height - new_height) // 2
        else:
            # Center on detected torso
            x_offset = center_x - new_width // 2
            
            # Adjust vertical position to align with shoulders
            left_shoulder, right_shoulder = torso_region['shoulder_points']
            y_shoulder = (left_shoulder[1] + right_shoulder[1]) // 2
            
            # Place top of outfit slightly above shoulders
            y_offset = y_shoulder - int(new_height * 0.15)  # 15% above shoulders
            
            # Fix for rotation issue - normalize the angle
            if rotation_angle < -90:
                rotation_angle += 180
            elif rotation_angle > 90:
                rotation_angle -= 180
        
        # Ensure offsets are within image bounds
        x_offset = max(0, min(x_offset, h_width - new_width))
        y_offset = max(0, min(y_offset, h_height - new_height))
        
        print(f"Placing outfit at position ({x_offset}, {y_offset})")
        
        # Rotate outfit if needed
        if abs(rotation_angle) > 1.0 and torso_region is not None:
            # Create a rotation matrix
            center = (new_width // 2, new_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Apply rotation to outfit image (preserve alpha channel)
            rotated_outfit = cv2.warpAffine(outfit_img, rotation_matrix, (new_width, new_height), 
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            outfit_img = rotated_outfit
        
        # Create a region of interest (ROI)
        roi_width = min(new_width, h_width - x_offset)
        roi_height = min(new_height, h_height - y_offset)
        
        if roi_width <= 0 or roi_height <= 0:
            print("ERROR: No valid region for overlay")
            return False
        
        # Get the region from human image
        roi = human_img[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width].copy()
        
        # Get the corresponding outfit region
        outfit_roi = outfit_img[0:roi_height, 0:roi_width]
        
        # Extract alpha channel and normalize to [0,1]
        alpha = outfit_roi[:, :, 3] / 255.0
        
        # Create a 3-channel alpha for blending
        alpha_3channel = np.stack([alpha, alpha, alpha], axis=2)
        
        # Apply segmentation-based adjustment if available (with fix for shape mismatch)
        if landmarks.segmentation_mask is not None:
            segmentation = landmarks.segmentation_mask
            segmentation = cv2.resize(segmentation, (h_width, h_height))
            roi_segmentation = segmentation[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width]
            
            # Use segmentation to adjust alpha blending (properly shaped for multiplication)
            segment_factor = np.expand_dims(roi_segmentation, axis=2)  # Make it (h,w,1)
            segment_factor = np.clip(segment_factor, 0.5, 1.0)  # Limit the adjustment
            
            # Apply adjustment to the alpha_3channel directly
            alpha_3channel = alpha_3channel * segment_factor
        
        # Blend the images using alpha channel
        blended_roi = roi * (1 - alpha_3channel) + outfit_roi[:, :, :3] * alpha_3channel
        
        # Place the blended region back into the human image
        result = human_img.copy()
        result[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width] = blended_roi
        
        # Save the result
        try:
            cv2.imwrite(output_path, result)
            print(f"Result saved to {output_path}")
            return True
        except Exception as e:
            print(f"ERROR saving result: {e}")
            return False

# Example usage
if __name__ == "__main__":
    fitting_room = VirtualFittingRoom()
    
    human_img_path = "person.jpg"  # Update this path
    outfit_img_path = "outfit.png"  # Update this path
    output_img_path = "result.jpg"  # Update this path
    
    success = fitting_room.overlay_outfit(human_img_path, outfit_img_path, output_img_path)
    if success:
        print("Outfit overlay completed successfully")
    else:
        print("Outfit overlay failed")