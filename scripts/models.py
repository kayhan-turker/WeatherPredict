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
        self.fc = nn.Linear(latent_dim + num_labels, 256 * H_DIV_16 * W_DIV_16)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.film1 = FiLMLayer(128, num_labels)
        self.film2 = FiLMLayer(64, num_labels)
        self.film3 = FiLMLayer(32, num_labels)

        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(32)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, latent, labels, return_features=False):
        z = torch.cat((latent, labels), dim=1)
        x = self.fc(z).view(-1, 256, H_DIV_16, W_DIV_16)
        f1 = self.leaky_relu(self.norm1(self.film1(self.conv1(x), labels)))
        f2 = self.leaky_relu(self.norm2(self.film2(self.conv2(f1), labels)))
        f3 = self.leaky_relu(self.norm3(self.film3(self.conv3(f2), labels)))
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

