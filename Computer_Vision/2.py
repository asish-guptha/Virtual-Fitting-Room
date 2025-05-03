import cv2
import sys
import threading
import numpy as np
from tkinter import filedialog
from rembg import remove

try:
    import tkinter as tk
    from tkinter import Button, Label
    from PIL import Image, ImageTk
except ModuleNotFoundError as e:
    print("Required module not found:", e)
    print("Ensure you have Tkinter and Pillow installed.")
    raise

# Initialize the camera
cap = cv2.VideoCapture(0)
selected_outfit = None
outfit_img = None

if not cap.isOpened():
    print("Error: Camera not accessible.")
    sys.exit(1)

def select_outfit(outfit_type):
    global selected_outfit
    selected_outfit = outfit_type
    print(f"Selected Outfit: {outfit_type}")

def remove_background(img):
    if img is None:
        print("Error: Invalid image for background removal.")
        return img
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = remove(img)  # Automatic background removal using rembg
    img = np.array(img)
    
    # Ensure proper transparency handling
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    return img

def upload_outfit():
    global outfit_img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Error: Unable to read the image file.")
            return
        img = remove_background(img)
        outfit_img = img
        print("Outfit uploaded successfully!")

def detect_body_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    return None

def overlay_outfit(frame):
    global outfit_img
    if outfit_img is not None:
        body_rect = detect_body_contour(frame)
        if body_rect:
            x, y, w, h = body_rect
            resized_outfit = cv2.resize(outfit_img, (w, int(h * 0.6)))  # Fit to upper body
            
            if resized_outfit.shape[-1] == 4:  # Handle transparency
                alpha = resized_outfit[:, :, 3] / 255.0
                for c in range(3):
                    frame[y:y+resized_outfit.shape[0], x:x+resized_outfit.shape[1], c] = (
                        (1 - alpha) * frame[y:y+resized_outfit.shape[0], x:x+resized_outfit.shape[1], c] +
                        alpha * resized_outfit[:, :, c]
                    )
            else:
                frame[y:y+resized_outfit.shape[0], x:x+resized_outfit.shape[1]] = resized_outfit
    return frame

def update_camera():
    global cap
    
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = overlay_outfit(frame)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    
    camera_label.after(10, update_camera)

# Setup Tkinter window
root = tk.Tk()
root.title("Virtual Fitting Room")
root.geometry("800x600")

# Camera display label
camera_label = Label(root)
camera_label.pack()

# Outfit selection buttons
button_frame = tk.Frame(root)
button_frame.pack()

shirt_button = Button(button_frame, text="Shirt", command=lambda: select_outfit("Shirt"))
shirt_button.pack(side=tk.LEFT, padx=10)

tshirt_button = Button(button_frame, text="T-Shirt", command=lambda: select_outfit("T-Shirt"))
tshirt_button.pack(side=tk.LEFT, padx=10)

pants_button = Button(button_frame, text="Pants", command=lambda: select_outfit("Pants"))
pants_button.pack(side=tk.LEFT, padx=10)

upload_button = Button(root, text="Upload Outfit", command=upload_outfit)
upload_button.pack()

# Start camera thread
threading.Thread(target=update_camera, daemon=True).start()

# Run Tkinter
try:
    root.mainloop()
finally:
    cap.release()
    cv2.destroyAllWindows()
