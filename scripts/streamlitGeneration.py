import streamlit as st
import torch
import numpy as np
from PIL import Image

# Load your trained model
model = torch.load("model.pth").eval()

st.title("GAN Image Generator")

# Sliders for latent vector input
latent_vector = np.array([st.slider(f"Z{i}", -1.0, 1.0, 0.0) for i in range(100)])
latent_tensor = torch.tensor(latent_vector, dtype=torch.float32).unsqueeze(0)

# Generate and display image
with torch.no_grad():
    img = model(latent_tensor).squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 127.5 + 127.5).astype(np.uint8)
    st.image(img)