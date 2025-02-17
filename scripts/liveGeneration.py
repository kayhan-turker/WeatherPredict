import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
import os

from models import *  # Import the generator class
from settings import *  # Ensure the settings match

# Load the trained generator
model_path = MODEL_SAVE_PATH + "2025_02_17_00_08_12_gen_epoch_141.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)

generator = FakeImageGenerator(LATENT_DIM, NUM_LABELS).to(device)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
generator.load_state_dict(checkpoint["state_dict"])
generator.label_means = checkpoint["label_means"]
generator.label_stds = checkpoint["label_stds"]
generator.eval()

output_folder = GENERATION_OUTPUT_PATH + model_path[:19] + "/"
os.makedirs(output_folder, exist_ok=True)

# Initialize latent vector
latent_vector = torch.randn(1, LATENT_DIM, device=device)
label_vector = torch.randn(1, NUM_LABELS, device=device)


# Function to update latent vector & regenerate image
def update_latent(index, val):
    global latent_vector
    if 0 <= index < LATENT_DIM:
        latent_vector[0, index] = float(val)
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
root = tk.Tk()
root.title("Latent Vector Controls")
root.geometry("350x600")  # Adjust width if needed

# Scrollable frame setup
container = ttk.Frame(root)
canvas = tk.Canvas(container)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

canvas.configure(yscrollcommand=scrollbar.set)

# Pack everything
container.pack(fill="both", expand=True)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Configure scrolling
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Create sliders
for i in range(LATENT_DIM):
    frame = ttk.Frame(scrollable_frame)
    frame.pack(fill="x", padx=5, pady=2)

    label = ttk.Label(frame, text=f"Z{i}:")
    label.pack(side="left")

    slider = ttk.Scale(frame, from_=-1, to=1, orient="horizontal",
                       command=lambda val, idx=i: update_latent(idx, val))
    slider.pack(side="right", fill="x", expand=True)


# Enable scrolling with mouse wheel
def on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")


root.bind_all("<MouseWheel>", on_mousewheel)

# Show initial image
update_image()

# Run Tkinter main loop
root.mainloop()
