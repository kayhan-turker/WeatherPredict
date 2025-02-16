from PIL import Image
import numpy as np
import os

from models import *  # Import the generator class
from settings import *  # Ensure the settings match


# Load the trained generator
LOAD_EPOCH = 10
MODEL_NAME = ""
model_name = get_model_save_name(MODEL_NAME, LOAD_EPOCH)
model_path = MODEL_SAVE_PATH + model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)

generator = FakeImageGenerator(LATENT_DIM, NUM_LABELS).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

output_folder = GENERATION_OUTPUT_PATH + model_name + "/"
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
generated_img.save("generated_example.png")  # Save the image
