import torch
import torch.nn as nn
import numpy as np


class SRNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SRNet, self).__init__()

        # --- IMPROVED SRM INITIALIZATION (Claude's Version) ---
        # Layer 1: 64 Filters total. We will set ~12 of them to SRM, rest random.
        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        srm_weights = np.zeros((64, 1, 3, 3), dtype=np.float32)

        # Filters 0-3: Basic High-Pass (KV kernel) with noise variation
        base_kv = [[-1, 2, -1],
                   [2, -4, 2],
                   [-1, 2, -1]]
        for i in range(4):
            srm_weights[i, 0] = base_kv
            srm_weights[i, 0] += np.random.randn(3, 3) * 0.05  # Small jitter

        # Filters 4-7: Edge Detectors
        base_edge = [[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]]
        for i in range(4, 8):
            srm_weights[i, 0] = base_edge
            srm_weights[i, 0] += np.random.randn(3, 3) * 0.05

        # Filter 8: Horizontal Edge
        srm_weights[8, 0] = [[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]]

        # Filter 9: Vertical Edge
        srm_weights[9, 0] = [[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]

        # Filter 10: Square Kernel
        srm_weights[10, 0] = [[-1, 2, -1],
                              [2, -4, 2],
                              [-1, 2, -1]]

        # Normalize specific filters
        srm_weights[:11] = srm_weights[:11] / 4.0

        with torch.no_grad():
            # Copy specific SRM filters to first 11 slots
            self.layer1.weight[:11].copy_(torch.from_numpy(srm_weights[:11]))
            # Initialize the remaining 53 filters with Kaiming Normal (Standard Deep Learning)
            nn.init.kaiming_normal_(self.layer1.weight[11:], mode='fan_out', nonlinearity='relu')

        # Wrap in Sequential
        self.layer1 = nn.Sequential(
            self.layer1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Layer 2: Reduce to 16 feature maps
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Layers 3-7: Residual blocks without pooling
        self.layer3 = self._make_layer(16, 16, stride=1)
        self.layer4 = self._make_layer(16, 16, stride=1)
        self.layer5 = self._make_layer(16, 16, stride=1)
        self.layer6 = self._make_layer(16, 16, stride=1)
        self.layer7 = self._make_layer(16, 16, stride=1)

        # Layers 8-11: Residual blocks with pooling
        self.layer8 = self._make_layer(16, 16, stride=1, pool=True)
        self.layer9 = self._make_layer(16, 64, stride=1, pool=True)
        self.layer10 = self._make_layer(64, 128, stride=1, pool=True)
        self.layer11 = self._make_layer(128, 256, stride=1, pool=True)

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride=1, pool=False):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        return ResidualBlock(nn.Sequential(*layers), in_channels, out_channels, pool)

    def _initialize_weights(self):
        # Claude's Fix: Robust check to skip layer1
        for name, m in self.named_modules():
            if 'layer1' in name:
                continue  # Skip because we did it manually
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.global_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, main_path, in_channels, out_channels, pool=False):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.pool = pool
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        if pool:
            self.shortcut_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.main_path(x)
        identity = self.shortcut(identity)
        if self.pool:
            identity = self.shortcut_pool(identity)
        out = out + identity
        out = self.relu(out)
        return out