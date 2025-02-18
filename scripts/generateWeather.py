from PIL import Image
import numpy as np
import os

from models import *  # Import the generator class
from settings import *  # Ensure the settings match
from config import *  # Ensure the settings match


# Load the trained generator
model_path = MODEL_SAVE_PATH + "2025_02_17_00_08_12_gen_epoch_31.pth"

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


def generate_image(input_labels):
    z = torch.randn(1, LATENT_DIM, device=device)
    labels = torch.tensor(input_labels, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        fake_img = generator(z, labels).squeeze(0).cpu().numpy().transpose(1, 2, 0)

    fake_img = ((fake_img + 1) * 127.5).astype(np.uint8)  # Rescale
    return Image.fromarray(fake_img)


# Example: Generate image with custom labels
example_labels = [0.5, -0.2, 1.1, 0.3, -0.9, 0.4, -0.1, 0.2]  # Match the expected label count
generated_img = generate_image(example_labels)
generated_img.show()  # View the image

output_name = get_image_output_name(example_labels, generator.label_means, generator.label_stds)
print("Saved: " + output_folder + "/" + output_name + ".jpg")
generated_img.save(output_folder + "/" + output_name + ".jpg")  # Save the image
