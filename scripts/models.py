import torch
import torch.nn as nn
import torch.nn.init as init

from config import *
from settings import *


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class NoiseInjection(nn.Module):
    def __init__(self, channels, weight=0.1):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1) * weight)  # Learnable weight for noise injection
        self.noise = None  # Placeholder for the constant noise during evaluation
        self.channels = channels

    def forward(self, x):
        if self.training:
            # During training, generate new noise each forward pass
            self.noise = torch.randn_like(x)
        else:
            # During evaluation, use the same noise generated in the first forward pass
            if self.noise is None or self.noise.size() != x.size():
                self.noise = torch.randn_like(x)

        return x + self.noise * self.weight


class FiLMLayer(nn.Module):
    def __init__(self, num_channels, num_labels):
        super().__init__()
        self.gamma = nn.Linear(num_labels, num_channels)
        self.beta = nn.Linear(num_labels, num_channels)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.gamma.weight)
        nn.init.constant_(self.gamma.bias, 1)  # Start gamma at 1 (identity scaling)
        nn.init.xavier_uniform_(self.beta.weight)
        nn.init.constant_(self.beta.bias, 0)  # Start beta at 0 (no shift)

    def forward(self, x, channels):
        gamma = self.gamma(channels).unsqueeze(2).unsqueeze(3)
        beta = self.beta(channels).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

    def get_std_parameters(self):
        return torch.cat([
            self.gamma.weight.std(dim=1, keepdim=True),
            self.gamma.bias.std(keepdim=True).unsqueeze(0),
            self.beta.weight.std(dim=1, keepdim=True),
            self.beta.bias.std(keepdim=True).unsqueeze(0)
        ])


class FakeImageGenerator(nn.Module):
    def __init__(self, latent_dim, num_labels):
        super(FakeImageGenerator, self).__init__()
        self.fc = nn.Linear(latent_dim + num_labels, 128 * H_DIV_16 * W_DIV_16)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(16)

        self.noise1 = NoiseInjection(64, 0.1)
        self.noise2 = NoiseInjection(32, 0.1)
        self.noise3 = NoiseInjection(16, 0.1)

        self.film_y1 = FiLMLayer(64, num_labels)
        self.film_y2 = FiLMLayer(32, num_labels)
        self.film_y3 = FiLMLayer(16, num_labels)

        self.film_z1 = FiLMLayer(64, latent_dim)
        self.film_z2 = FiLMLayer(32, latent_dim)
        self.film_z3 = FiLMLayer(16, latent_dim)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, y, z, return_features=False):
        x = torch.cat((y, z), dim=1)
        x = self.fc(x).view(-1, 128, H_DIV_16, W_DIV_16)

        f1 = self.noise1(self.norm1(self.conv1(x)))
        f1 = self.leaky_relu(self.film_z1(self.film_y1(f1, y), z))

        f2 = self.noise2(self.norm2(self.conv2(f1)))
        f2 = self.leaky_relu(self.film_z2(self.film_y2(f2, y), z))

        f3 = self.noise3(self.norm3(self.conv3(f2)))
        f3 = self.leaky_relu(self.film_z3(self.film_y3(f3, y), z))

        x = self.tanh(self.conv4(f3))
        if return_features:
            return x, [f1, f2, f3]
        return x

    def save_model(self, model_save_name, label_means, label_stds):
        save_path = MODEL_SAVE_PATH + model_save_name
        torch.save({
            "model": self,
            "label_means": label_means,
            "label_stds": label_stds
        }, save_path)
        print("\n" + "=" * 100)
        print(f"Saved Generator Model: {save_path}")
        print("\n" + "=" * 100)


class LabelPredictor(nn.Module):
    def __init__(self, output_size):
        super(LabelPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * H_DIV_8 * W_DIV_8, 128)
        self.fc2 = nn.Linear(128, output_size + 1)  # Updated to match new label size

        self.pool = nn.MaxPool2d(2, 2)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.norm1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.norm2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)


def get_model_save_name(model_name, epoch):
    return f"{model_name}_gen_epoch_{epoch + 1}.pth"

