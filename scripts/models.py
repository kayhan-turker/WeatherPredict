import torch
import torch.nn as nn

from settings import *


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if isinstance(m, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            else:
                nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


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
        x = self.norm(x)
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
        self.fc_labels = nn.Linear(num_labels, 64 * 8 * 16)
        self.fc_latent = nn.Linear(latent_dim, 64 * 8 * 16)

        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=2)

        # self.norm1 = nn.LayerNorm([num_channels, 1, 1], elementwise_affine=False)
        # self.norm2 = nn.LayerNorm([num_channels, 1, 1], elementwise_affine=False)
        # self.norm3 = nn.LayerNorm([num_channels, 1, 1], elementwise_affine=False)

        self.fc_film = nn.Linear(64 * 8 * 16, 128)

        self.film1 = FiLMLayer(32, 128)
        self.film2 = FiLMLayer(16, 128)
        self.film3 = FiLMLayer(8, 128)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, latent, labels):
        z_labels = self.fc_labels(labels)
        z_latent = self.fc_latent(latent)
        z = z_labels + z_latent

        x = z.view(-1, 64, 8, 16)
        z_film = self.fc_film(z)
        x = self.upsample(x)
        x = self.leaky_relu(self.film1(self.conv1(x), z_film))
        x = self.upsample(x)
        x = self.leaky_relu(self.film2(self.conv2(x), z_film))
        x = self.upsample(x)
        x = self.leaky_relu(self.film3(self.conv3(x), z_film))
        x = self.upsample(x)
        x = self.tanh(self.conv4(x))
        return x


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


def get_model_save_name(model_name, epoch):
    return f"{model_name}_gen_epoch_{epoch + 1}.pth"


def get_image_output_name(labels, label_means, label_stds):
    if isinstance(labels, torch.Tensor):
        output_name = [x for x in labels[0].detach().cpu().numpy()]
    else:
        output_name = [x for x in labels]
    non_shift_labels = len(output_name) - len(SELECTED_INDICES)
    for x in range(non_shift_labels, len(output_name)):
        output_name[x] = output_name[x] * label_stds[x - non_shift_labels] + label_means[x - non_shift_labels]
    output_name[0] = output_name[0] / 2 + 0.5
    output_name[1] = output_name[1] / 2 + 0.5
    return "_".join(f"{output_name[x]:.2f}" if x < 2 else f"{int(round(output_name[x]))}" for x in range(len(output_name)))