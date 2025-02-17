import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import torch.backends.cudnn as cudnn

from models import *
from settings import *
from config import *

# ====================================
# 0. Preparation
# ====================================

prev_time = [time.perf_counter()]


# ====================================
# 1. Normalize Labels
# ====================================
def compute_label_stats():
    print("\n" + "=" * 100)
    print("1. Scan Files For Label Statistics")
    print("-" * 100)

    all_labels = []

    print(f"  Get label data from file names. (Prev step {get_elapsed_ms(prev_time):.2f} ms).")
    for filename in os.listdir(TRAINING_DATA_PATH):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            filename = os.path.splitext(filename)[0]  # Remove file extension
            fields = filename.split('_')

            numeric_values = []
            for i in SELECTED_INDICES:
                if i < len(fields):
                    try:
                        value = float(fields[i])
                        numeric_values.append(value)
                    except ValueError:
                        numeric_values.append(None)  # Mark non-numeric values as None

            if None not in numeric_values:
                all_labels.append(numeric_values)
            else:
                print_error(f"  Skipping {filename} due to missing/invalid numeric fields.")

    num_images = len(all_labels)
    if num_images == 0:
        raise ValueError("No valid labels found. Terminating training.")

    all_labels = np.array(all_labels, dtype=np.float32)

    print(f"  Calculating label statistics. (Prev step {get_elapsed_ms(prev_time):.2f} ms).")
    label_means = all_labels.mean(axis=0)
    label_stds = all_labels.std(axis=0, ddof=1)
    label_stds = np.where(label_stds == 0, 1, label_stds)  # Prevent division by zero

    print(f"    Label Means:               {label_means}")
    print(f"    Label Standard Deviations: {label_stds}")
    print(f"  Label statistics complete. (Prev step {get_elapsed_ms(prev_time):.2f} ms).")

    return num_images, label_means, label_stds


NUM_IMAGES, label_means, label_stds = compute_label_stats()


# ====================================
# 2. Normalize Dataset
# ====================================
class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        print(f"  Initialize image dataset. (Prev step {get_elapsed_ms(prev_time):.2f} ms).")
        self.folder = folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.images_loaded = 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        filename = os.path.splitext(os.path.basename(img_path))[0]
        fields = filename.split('_')
        fields = [safe_float(x) for x in fields]

        if any(fields[x] is None for x in SELECTED_INDICES) or fields[1] is None or fields[2] is None:
            print_error("  None value found when getting item labels. Skipping and retrieving next item.")
            return self.__getitem__((idx + 1) % len(self))

        value_1 = fields[1] / 12 + fields[2] / (31 * 12) if fields[1] is not None and fields[2] is not None else 0
        value_2 = (fields[3] / 24 + fields[4] / (24 * 60) + fields[5] / (24 * 60 * 60)
                   if fields[3] is not None and fields[4] is not None and fields[5] is not None else 0)

        numeric_labels = [fields[i] if fields[i] is not None else 0 for i in SELECTED_INDICES]
        standardized_labels = [(numeric_labels[i] - label_means[i]) / label_stds[i] for i in range(len(numeric_labels))]

        full_labels = torch.tensor([value_1, value_2] + standardized_labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        self.images_loaded += 1
        return image, full_labels


# ====================================
# 3. Transformations
# ====================================
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.Pad((14, 0, 15, 0), fill=0, padding_mode="constant"),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("\n" + "=" * 100)
print("2. Build Dataset")
print("-" * 100)
print("  Initialize dataset.")
dataset = ImageDataset(folder=TRAINING_DATA_PATH, transform=transform)

print("  Initialize data loader.")
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

NUM_BATCHES = NUM_IMAGES // BATCH_SIZE
print(f"  Num Images: {NUM_IMAGES}, Num Labels {NUM_LABELS}, Num Batches {NUM_BATCHES}")


# ====================================
# 7. GPU Setup
# ====================================

print("\n" + "=" * 100)
print("7. GPU Setup")
print("-" * 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)

print("  Using device:", device)
print("  CUDA available:", torch.cuda.is_available())  # Should print True
print("  Devices available:", torch.cuda.device_count())  # Should print number of GPUs available
if torch.cuda.is_available():
    print("  Current device index:", torch.cuda.current_device())  # Should print device index
    print("  CUDA device name:", torch.cuda.get_device_name(0))  # Should print your GPU name
    print("  cuDNN Enabled:", cudnn.enabled)
    print("  cuDNN Benchmark:", cudnn.benchmark)

# ====================================
# 8. Training Setup
# ====================================

print("\n" + "=" * 100)
print("8. Training Setup")
print("-" * 100)

print("  Initialize models.")
generator = FakeImageGenerator(LATENT_DIM, NUM_LABELS).to(device)
discriminator = LabelPredictor(NUM_LABELS).to(device)
print("  Apply weights.")
generator.apply(weights_init)
discriminator.apply(weights_init)

criterion_labels = nn.MSELoss()
criterion_realism = nn.BCEWithLogitsLoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

print("  Generator device:", next(generator.parameters()).device)  # Should print "cuda:0"
print("  Discriminator device:", next(discriminator.parameters()).device)  # Should print "cuda:0"

# ====================================
# 9. Training Loop
# ====================================

print("\n" + "=" * 100)
print("9. Begin Training")
print("-" * 100)

epochs = 150
last_epoch_time = datetime.now()
g_label_loss_factor = 4.0
d_label_loss_factor = 1.5
fake_repulsion = 0.2

# Create directory to store outputs
ts = datetime.now()
ts = ts.strftime("%Y_%m_%d_%H_%M_%S")
MODEL_NAME = ts
output_folder = GENERATION_OUTPUT_PATH + "training/" + MODEL_NAME
os.makedirs(output_folder, exist_ok=True)

for epoch in range(epochs):
    batch_num = 0
    for images, labels in trainloader:
        batch_num += 1
        print("", end="\r")
        print(f"Batch progress: {((batch_num * 100) // NUM_BATCHES):.1f}% [{('#' * (batch_num // 3))}"
              f"{('-' * ((NUM_BATCHES // 3 - batch_num // 3)))}]", end="")
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # ----------------------
        # Train Generator
        # ----------------------
        latent = torch.randn(labels.shape[0], LATENT_DIM).to(device, non_blocking=True)  # Random noise
        linear_part = torch.rand(labels.shape[0], 2).to(device, non_blocking=True) * 2 - 1
        normal_part = torch.randn(labels.shape[0], labels.shape[1] - 2).to(device, non_blocking=True)
        random_labels = torch.cat([linear_part, normal_part], dim=1)

        fake_images = generator(latent, random_labels)
        pred_fake = discriminator(fake_images)  # No manual modification

        loss_G_realism = criterion_realism(pred_fake[:, -1], torch.ones_like(pred_fake[:, -1], device=device))
        loss_G_labels = criterion_labels(pred_fake[:, :-1], random_labels)
        loss_G = loss_G_realism + g_label_loss_factor * loss_G_labels

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ----------------------
        # Train Discriminator
        # ----------------------
        pred_real = discriminator(images)
        pred_fake = discriminator(fake_images.detach())

        # Label consistency loss for discriminator
        loss_D_realism_real = criterion_realism(pred_real[:, -1], torch.ones_like(pred_real[:, -1], device=device))
        loss_D_realism_fake = criterion_realism(pred_fake[:, -1], torch.zeros_like(pred_fake[:, -1], device=device))
        loss_D_labels_real = criterion_labels(pred_real[:, :-1], labels)
        loss_D_labels_fake = criterion_labels(pred_fake[:, :-1], random_labels)

        loss_D = loss_D_realism_real + loss_D_realism_fake + d_label_loss_factor * (loss_D_labels_real + fake_repulsion / (loss_D_labels_fake + 1))

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    # Save model every 10 epochs
    if epoch % 10 == 0:
        save_path = MODEL_SAVE_PATH + get_model_save_name(MODEL_NAME, epoch)
        torch.save({
            "state_dict": generator.state_dict(),
            "label_means": label_means,
            "label_stds": label_stds
        }, save_path)
        print("\n" + "=" * 100)
        print(f"Saved Generator Model: {save_path}")
        print("\n" + "=" * 100)

    output_name = get_image_output_name(random_labels, label_means, label_stds)

    fake_image_np = ((fake_images[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    fake_image_pil = Image.fromarray(fake_image_np)
    fake_image_pil.save(f"{output_folder}/E{(epoch + 1):03}_{output_name}.png")  # Save fake image

    # Get debug information
    current_time = datetime.now()
    time_since_last = (current_time - last_epoch_time).total_seconds()
    datetime_next = current_time + (current_time - last_epoch_time)

    film1_mean_params = generator.film1.get_mean_parameters()
    film2_mean_params = generator.film2.get_mean_parameters()
    film3_mean_params = generator.film3.get_mean_parameters()

    # Print details
    print("\n" + "=" * 100)
    print(f"Time: {str(current_time)[:19]} | Epoch Time (s): {time_since_last:.2f} | Next Epoch: {str(datetime_next)[:19]}")
    print("-" * 100)
    print(f"Epoch {epoch + 1}/{epochs}, Loss_G: {loss_G.item():.3f}, Loss_D: {loss_D.item():.3f}")
    print("")
    print(f"G Realism Score: {torch.abs(torch.ones_like(pred_fake[:, -1]) - pred_fake[:, -1]).mean().item():.3f} "
          f"Label Score: [{', '.join([f'{torch.abs(random_labels[:, x] - pred_fake[:, x]).mean().item():.3f}' for x in range(NUM_LABELS)])}]")
    print(f"D Realism Score: {torch.abs(torch.ones_like(pred_real[:, -1]) - pred_real[:, -1]).mean().item():.3f} "
          f"Label Score: [{', '.join([f'{torch.abs(labels[:, x] - pred_real[:, x]).mean().item():.3f}' for x in range(NUM_LABELS)])}]")
    print("")
    print(f"G FiLM Gamma Weights: γ1 = {film1_mean_params[0]:3f}, γ2 = {film2_mean_params[0]:3f}, γ3 = {film3_mean_params[0]:3f}")
    print(f"G FiLM Gamma Biases: γ1 = {film1_mean_params[1]:3f}, γ2 = {film2_mean_params[1]:3f}, γ3 = {film3_mean_params[1]:3f}")
    print(f"G FiLM Beta Weights: β1 = {film1_mean_params[2]:3f}, β2 = {film2_mean_params[2]:3f}, β3 = {film3_mean_params[2]:3f}")
    print(f"G FiLM Beta Biases: β1 = {film1_mean_params[3]:3f}, β2 = {film2_mean_params[3]:3f}, β3 = {film3_mean_params[3]:3f}")
    print("")
    print("                                  ['Date', 'Time', 'Temp', 'Press', 'Dew', 'Hum', 'Dir', 'Alt']")
    print("Real Input Labels:               ", [f"{x:.1f}" for x in labels[0].cpu().numpy()])
    print("Predicted Labels (Real):         ", [f"{x:.1f}" for x in pred_real[0, :-1].detach().cpu().numpy()])
    print("Random Labels (Generator Input): ", [f"{x:.1f}" for x in random_labels[0].cpu().numpy()])
    print("Predicted Labels (Fake):         ", [f"{x:.1f}" for x in pred_fake[0, :-1].detach().cpu().numpy()])
    print(f"Real Check (Real): {pred_real[0, -1].item():.1f} | (Fake): {pred_fake[0, -1].item():.1f}")
    print("-" * 100)
    print("")

    last_epoch_time = datetime.now()

print("Training complete!")
