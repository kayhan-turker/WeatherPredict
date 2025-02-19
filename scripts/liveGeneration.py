import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk

from models import *
from settings import *
from config import *

# Load the trained generator
model_path = MODEL_SAVE_PATH + "model_034a.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)

checkpoint = torch.load(model_path, map_location=device)
generator = checkpoint["model"].to(device)
generator.eval()

label_means = checkpoint["label_means"]
label_stds = checkpoint["label_stds"]

# Initialize latent vector
latent_vector = torch.randn(1, LATENT_DIM, device=device)
label_vector = torch.randn(1, NUM_LABELS, device=device)


# Function to update latent vector & regenerate image
def update_latent(index, val):
    global latent_vector
    if 0 <= index < LATENT_DIM:
        latent_vector[0, index] = float(val)
        update_image()


def update_label(index, val):
    global latent_vector
    if 0 <= index < NUM_LABELS:
        label_vector[0, index] = float(val)
        update_image()


# Function to generate and display an image
def update_image():
    with torch.no_grad():
        img_tensor = generator(latent_vector, label_vector).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_tensor = ((img_tensor + 1) * 127.5).astype(np.uint8)  # Convert to 0-255 range
        img_tensor = cv2.resize(img_tensor, (512, 256), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Generated Image", img_tensor)
    cv2.waitKey(1)


# Tkinter GUI
window_width, window_height = 150, 400
vector_window = tk.Tk()
vector_window.title("Vector Controls")
vector_window.geometry(f"{window_width}x{window_height}")  # Adjust width if needed


# Scrollable frame setup
main_frame = ttk.Frame(vector_window)
main_frame.pack(fill="both", expand=True)
main_frame.grid_rowconfigure(0, weight=2)
main_frame.grid_rowconfigure(1, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

label_scrolling_canvas = tk.Canvas(main_frame, bg="#DDEEDD")
label_scrolling_canvas.grid(row=0, column=0, sticky="nsew")
label_scrollbar = ttk.Scrollbar(label_scrolling_canvas, orient="vertical", command=label_scrolling_canvas.yview)
label_scrolling_canvas.configure(yscrollcommand=label_scrollbar.set)

latent_scrolling_canvas = tk.Canvas(main_frame, bg="#DDDDEE")
latent_scrolling_canvas.grid(row=1, column=0, sticky="nsew")
latent_scrollbar = ttk.Scrollbar(latent_scrolling_canvas, orient="vertical", command=latent_scrolling_canvas.yview)
latent_scrolling_canvas.configure(yscrollcommand=latent_scrollbar.set)

label_slider_frame = ttk.Frame(label_scrolling_canvas)
latent_slider_frame = ttk.Frame(latent_scrolling_canvas)

# Pack everything
main_frame.pack(fill="both", expand=True)
label_scrollbar.pack(side="right", fill="y")
latent_scrollbar.pack(side="right", fill="y")

# Configure scrolling
label_scrolling_canvas.bind("<Configure>", lambda e: label_scrolling_canvas.configure(scrollregion=label_scrolling_canvas.bbox("all")))
latent_scrolling_canvas.bind("<Configure>", lambda e: latent_scrolling_canvas.configure(scrollregion=latent_scrolling_canvas.bbox("all")))
window_label = label_scrolling_canvas.create_window((0, 0), window=label_slider_frame, anchor="nw")
window_latent = latent_scrolling_canvas.create_window((1, 0), window=latent_slider_frame, anchor="nw")

# Create sliders
for i in range(NUM_LABELS):
    frame = ttk.Frame(label_slider_frame)
    frame.pack(fill="x", padx=5, pady=2)
    label = ttk.Label(frame, text=f"{LABEL_NAMES[i]}:")
    label.pack(side="left")
    slider = ttk.Scale(frame, from_=-1, to=1, orient="horizontal", command=lambda val, idx=i: update_label(idx, val))
    slider.pack(side="right", fill="x", expand=True)

for i in range(LATENT_DIM):
    frame = ttk.Frame(latent_slider_frame)
    frame.pack(fill="x", padx=5, pady=2)
    label = ttk.Label(frame, text=f"Z{i}:")
    label.pack(side="left")
    slider = ttk.Scale(frame, from_=-1, to=1, orient="horizontal", command=lambda val, idx=i: update_latent(idx, val))
    slider.pack(side="right", fill="x", expand=True)


# Enable scrolling with mouse wheel
def on_mousewheel(event):
    latent_scrolling_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


vector_window.bind_all("<MouseWheel>", on_mousewheel)

# Show initial image
update_image()

# Run Tkinter main loop
vector_window.mainloop()
