import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import torch.backends.cudnn as cudnn

from models import *
from config import *
from trainingResults import *

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
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
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

num_epochs = 150
last_epoch_time = datetime.now()
g_label_loss_factor = 8.0
g_latent_feature_shift = 0.3 / 3
d_latent_shift_loss = 1.0
d_label_loss_factor = 1.5

# Create directory to store outputs
ts = datetime.now()
ts = ts.strftime("%Y_%m_%d_%H_%M_%S")
MODEL_NAME = ts
output_folder = GENERATION_OUTPUT_PATH + "training/" + MODEL_NAME
os.makedirs(output_folder, exist_ok=True)

for epoch in range(num_epochs):
    batch_num = 0
    for images, labels in trainloader:
        batch_num += 1
        print("", end="\r")
        print(f"Batch progress: {((batch_num * 100) // NUM_BATCHES):.1f}% [{('#' * round(batch_num / NUM_BATCHES * PROGRESS_BAR_SIZE))}"
              f"{('-' * round((1 - batch_num / NUM_BATCHES) * PROGRESS_BAR_SIZE))}]", end="")
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # ----------------------
        # Train Generator
        # ----------------------
        x_in = torch.randn(labels.shape[0], IMAGE_HEIGHT * IMAGE_WIDTH).to(device, non_blocking=True)
        latent = torch.randn(labels.shape[0], LATENT_DIM).to(device, non_blocking=True)
        linear_part = torch.rand(labels.shape[0], 2).to(device, non_blocking=True) * 2 - 1
        normal_part = torch.randn(labels.shape[0], labels.shape[1] - 2).to(device, non_blocking=True)
        fake_labels = torch.cat([linear_part, normal_part], dim=1)

        fake_images, features = generator(fake_labels, latent, return_features=True)
        pred_fake = discriminator(fake_images)

        # Generate another image to analyze latent shift. Use the same labels
        latent_shifted = torch.randn(labels.shape[0], LATENT_DIM).to(device, non_blocking=True)
        fake_images_shifted, features_shifted = generator(fake_labels, latent, return_features=True)
        pred_fake_shifted = discriminator(fake_images)

        delta_latent = torch.mean(torch.abs(latent_shifted - latent), dim=1)
        delta_feature = sum(torch.mean(torch.abs(f1 - f2), dim=[1, 2, 3]) for f1, f2 in zip(features, features_shifted))

        # Calculate loss
        loss_G_realism = criterion_realism(pred_fake[:, -1], torch.ones_like(pred_fake[:, -1], device=device))
        loss_G_labels = criterion_labels(pred_fake[:, :-1], fake_labels)
        loss_G_latent_response = criterion_labels(delta_latent, g_latent_feature_shift * delta_feature)

        loss_G = loss_G_realism + g_label_loss_factor * loss_G_labels + d_latent_shift_loss * loss_G_latent_response

        # Back propagate
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
        loss_D_labels_fake = criterion_labels(pred_fake[:, :-1], fake_labels)

        loss_D = loss_D_realism_real + loss_D_realism_fake + d_label_loss_factor * loss_D_labels_real

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    # Save model every 10 epochs
    if epoch % 10 == 0:
        generator.save_model(get_model_save_name(MODEL_NAME, epoch), label_means, label_stds)

    save_generator_image(fake_images[0], output_folder, epoch, fake_labels, label_means, label_stds)

    refresh_results(generator, epoch, num_epochs, last_epoch_time, loss_G, loss_D,
                    labels, pred_real, fake_labels, pred_fake, delta_latent, delta_feature)
    last_epoch_time = datetime.now()

print("Training complete!")
