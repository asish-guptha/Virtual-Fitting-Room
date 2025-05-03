import cv2
import numpy as np
import os

def simple_outfit_overlay(human_image_path, outfit_image_path, output_path):
    """
    A simplified approach to overlay an outfit onto a human image.
    This doesn't rely on keypoints but allows manual positioning.
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
    
    # Simple scaling to fit outfit within human image
    # Scale down outfit if it's larger than 80% of the human image
    scale_x = min(1.0, (h_width * 0.8) / o_width)
    scale_y = min(1.0, (h_height * 0.8) / o_height)
    scale = min(scale_x, scale_y)
    
    # Only resize if we need to scale down
    if scale < 1.0:
        new_width = int(o_width * scale)
        new_height = int(o_height * scale)
        print(f"Resizing outfit from {o_width}x{o_height} to {new_width}x{new_height}")
        outfit_img = cv2.resize(outfit_img, (new_width, new_height))
        o_height, o_width = new_height, new_width
    
    # Calculate center position - place outfit in the middle of the human image
    x_offset = (h_width - o_width) // 2
    y_offset = (h_height - o_height) // 2
    
    # Ensure offsets are positive (outfit within image bounds)
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)
    
    print(f"Placing outfit at position ({x_offset}, {y_offset})")
    
    # Create a region of interest (ROI)
    roi_width = min(o_width, h_width - x_offset)
    roi_height = min(o_height, h_height - y_offset)
    
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
human_img_path = "person.jpg"  # Update this path
outfit_img_path = "outfit.png"  # Update this path
output_img_path = "result.jpg"  # Update this path

success = simple_outfit_overlay(human_img_path, outfit_img_path, output_img_path)
if success:
    print("Overlay completed successfully")
else:
    print("Overlay failed")