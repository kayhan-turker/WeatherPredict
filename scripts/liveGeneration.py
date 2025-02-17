import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
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

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)


# Function to generate image
def generate_image():
    with torch.no_grad():
        img = generator(latent_vector, label_vector).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize
        return img


# Display first image
image_display = ax.imshow(generate_image())

# Create sliders
axcolor = 'lightgoldenrodyellow'
slider_axes = [plt.axes([0.2, 0.0 + i * 0.03, 0.65, 0.02], facecolor=axcolor) for i in range(NUM_LABELS)]
sliders = [Slider(ax, f'Z{i}', -1.0, 1.0, valinit=latent_vector[0, i].item()) for i, ax in enumerate(slider_axes)]


# Update function
def update(val):
    for i, slider in enumerate(sliders):
        label_vector[0, i] = slider.val
    image_display.set_data(generate_image())
    fig.canvas.draw_idle()


# Connect sliders to update function
for slider in sliders:
    slider.on_changed(update)

plt.show()
