import torch
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime

from config import *


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        if isinstance(m, nn.ConvTranspose2d) and isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')  # Best for LeakyReLU
        else:
            init.xavier_uniform_(m.weight)  # Best for Tanh/Sigmoid
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class FiLMLayer(nn.Module):
    def __init__(self, num_features, num_labels):
        super().__init__()
        self.gamma = nn.Linear(num_labels, num_features)
        self.beta = nn.Linear(num_labels, num_features)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.gamma.weight)
        nn.init.constant_(self.gamma.bias, 1)  # Start gamma at 1 (identity scaling)
        nn.init.xavier_uniform_(self.beta.weight)
        nn.init.constant_(self.beta.bias, 0)  # Start beta at 0 (no shift)

    def forward(self, x, labels):
        gamma = self.gamma(labels).unsqueeze(2).unsqueeze(3)
        beta = self.beta(labels).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta  # Scale and shift feature maps

    def get_mean_parameters(self):
        gamma_vals_weights = self.gamma.weight.mean().item()
        gamma_vals_biases = self.gamma.bias.mean().item()
        beta_vals_weights = self.beta.weight.mean().item()
        beta_vals_biases = self.beta.bias.mean().item()
        return [gamma_vals_weights, gamma_vals_biases, beta_vals_weights, beta_vals_biases]


class FakeImageGenerator(nn.Module):
    def __init__(self, latent_dim, num_labels):
        super(FakeImageGenerator, self).__init__()
        self.fc = nn.Linear(latent_dim + num_labels, 256 * 8 * 16)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

        self.film1 = FiLMLayer(128, num_labels)
        self.film2 = FiLMLayer(64, num_labels)
        self.film3 = FiLMLayer(32, num_labels)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        z = torch.cat((z, labels), dim=1)
        x = self.fc(z).view(-1, 256, 8, 16)
        x = self.leaky_relu(self.film1(self.conv1(x), labels))
        x = self.leaky_relu(self.film2(self.conv2(x), labels))
        x = self.leaky_relu(self.film3(self.conv3(x), labels))
        x = self.tanh(self.conv4(x))
        return x  # Output is (batch_size, 3, 128, 256)


class LabelPredictor(nn.Module):
    def __init__(self, output_size):
        super(LabelPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 32, 128)
        self.fc2 = nn.Linear(128, output_size + 1)  # Updated to match new label size

        self.pool = nn.MaxPool2d(2, 2)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.pool(self.leaky_relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)


def get_model_save_name(epoch):
    ts = datetime.now()
    ts = ts.strftime("%Y_%m_%d_%H_%M_%S")
    return f"{ts}_gen_epoch_{epoch + 1}.pth"
