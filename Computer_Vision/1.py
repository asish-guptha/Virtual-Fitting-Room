import cv2
import sys
import threading
import numpy as np

try:
    import tkinter as tk
    from tkinter import Button
    from PIL import Image, ImageTk, ImageDraw
except ModuleNotFoundError as e:
    print("Required module not found:", e)
    print("Ensure you have Tkinter and Pillow installed.")
    raise

# Initialize the camera
cap = cv2.VideoCapture(0)
selected_outfit = None

if not cap.isOpened():
    print("Error: Camera not accessible.")
    sys.exit(1)

def select_outfit(outfit_type):
    global selected_outfit
    selected_outfit = outfit_type
    print(f"Selected Outfit: {outfit_type}")

def draw_body_outline(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection
    frame[edges != 0] = [0, 255, 0]  # Green outline
    return frame

def update_camera():
    global cap, selected_outfit
    
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)  # Mirror effect
        frame = draw_body_outline(frame)  # Draw detected body outline
        
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
camera_label = tk.Label(root)
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

# Start camera thread
threading.Thread(target=update_camera, daemon=True).start()

# Run Tkinter
try:
    root.mainloop()
finally:
    # Release the camera when closing
    cap.release()
    cv2.destroyAllWindows()
