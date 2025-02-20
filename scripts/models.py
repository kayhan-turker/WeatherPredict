import torch
import torch.nn as nn

from settings import *


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


def init_conv_transpose_weights(m):
    if isinstance(m, nn.ConvTranspose2d):
        print("convtrans2d init")
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_conv_weights(m):
    if isinstance(m, nn.Conv2d):
        print("conv2d init")
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_fc_weights(m):
    if isinstance(m, nn.Linear):
        print("linear init")
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_bn_weights(m):
    if isinstance(m, nn.BatchNorm2d):
        print("batch norm init")
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def init_film_weights(m):
    if isinstance(m, FiLMLayer):
        print("film init")
        nn.init.kaiming_normal_(m.gamma.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(m.gamma.bias, 0)
        nn.init.kaiming_normal_(m.beta.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(m.beta.bias, 0)


def init_weights(m):
    init_conv_weights(m)
    init_conv_transpose_weights(m)
    init_fc_weights(m)
    init_bn_weights(m)
    init_film_weights(m)


class FakeImageGenerator(nn.Module):
    def __init__(self, y_dim, z_dim):
        super(FakeImageGenerator, self).__init__()
        self.fc_y = nn.Linear(y_dim, 64 * H_DIV_16 * W_DIV_16)
        self.fc_z = nn.Linear(z_dim, 64 * H_DIV_16 * W_DIV_16)

        self.conv_x1 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.conv_x2 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
        self.conv_x3 = nn.ConvTranspose2d(24, 8, kernel_size=4, stride=2, padding=1)
        self.conv_x4 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

        self.conv_y1 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
        self.conv_y2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)
        self.conv_y3 = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=1)

        self.conv_z1 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
        self.conv_z2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)
        self.conv_z3 = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=1)

        self.noise1 = NoiseInjection(64, 0.002)
        self.noise2 = NoiseInjection(24, 0.004)
        self.noise3 = NoiseInjection(16, 0.006)

        self.film1 = FiLMLayer(64, y_dim)
        self.film2 = FiLMLayer(24, y_dim)
        self.film3 = FiLMLayer(16, y_dim)

        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(24)
        self.norm3 = nn.BatchNorm2d(16)

        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, y, z, return_features=False):
        y_init = torch.cat((y, z), dim=1)
        z = self.fc_z(z).view(-1, 64, H_DIV_16, W_DIV_16)         # -> 64 x W/16 x H/16
        y = self.fc_y(y).view(-1, 64, H_DIV_16, W_DIV_16)         # -> 64 x W/16 x H/16
        x = torch.cat((y, z), dim=1)                     # -> 128 x W/16 x H/16

        y = self.conv_y1(y)                                     # 64 x W/16 x H/16 -> 16 x W/8 x H/8
        z = self.conv_z1(z)                                     # 64 x W/16 x H/16 -> 16 x W/8 x H/8
        f1 = torch.cat((self.conv_x1(x), y, z), dim=1)   # 128 x W/16 x H/16 -> 32 x W/8 x H/8 -> 64 x W/8 x H/8
        f1 = self.leaky_relu(self.norm1(self.film1(self.noise1(f1), y_init)))

        y = self.conv_y2(y)                                     # 16 x W/8 x H/8 -> 4 x W/4 x H/4
        z = self.conv_z2(z)                                     # 16 x W/8 x H/8 -> 4 x W/4 x H/4
        f2 = torch.cat((self.conv_x2(f1), y, z), dim=1)  # 64 x W/8 x H/8 -> 16 x W/4 x H/4 -> 24 x W/4 x H/4
        f2 = self.leaky_relu(self.norm2(self.film2(self.noise2(f2), y_init)))

        y = self.conv_y3(y)                                     # 4 x W/4 x H/4 -> 4 x W/2 x H/2
        z = self.conv_z3(z)                                     # 4 x W/4 x H/4 -> 4 x W/2 x H/2
        f3 = torch.cat((self.conv_x3(f2), y, z), dim=1)  # 24 x W/4 x H/4 -> 8 x W/2 x H/2 -> 16 x W/2 x H/2
        f3 = self.leaky_relu(self.norm3(self.film3(self.noise3(f3), y_init)))

        x = self.tanh(self.conv_x4(f3))                         # 16 x W/2 x H/2 -> 3 x W/1 x H/1
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


class LabelPredictor(nn.Module):        # todo increase model size to pick up more detail/patterns for the generator to eventually learn
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

