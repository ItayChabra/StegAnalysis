import torch
import torch.nn as nn
import numpy as np


class SRNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SRNet, self).__init__()

        # --- TRIPLE-BRANCH FRONTEND (Run 13) ---

        # Branch A: 11 frozen SRM filters — reads spatial only (Channel 0)
        self.branch_a = nn.Conv2d(1, 11, kernel_size=3, stride=1, padding=1, bias=False)
        srm_weights = np.zeros((11, 1, 3, 3), dtype=np.float32)

        # Filters 0-3: KV kernel jitter
        base_kv = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        for i in range(4):
            srm_weights[i, 0] = base_kv + np.random.randn(3, 3) * 0.05

        # Filters 4-7: Edge Detectors jitter
        base_edge = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        for i in range(4, 8):
            srm_weights[i, 0] = base_edge + np.random.randn(3, 3) * 0.05
        srm_weights[8, 0] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]  # Horizontal
        srm_weights[9, 0] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # Vertical
        srm_weights[10, 0] = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]  # Square

        with torch.no_grad():
            self.branch_a.weight.copy_(torch.from_numpy(srm_weights / 4.0))
        self.branch_a.requires_grad_(False)
        self.bn_a = nn.BatchNorm2d(11)

        # Branch B: 53 learnable spatial filters — reads spatial only (Channel 0)
        # Purpose: Learn complex LSB/DCT patterns that rigid SRM misses
        self.branch_b = nn.Conv2d(1, 53, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(53)

        # Branch C: 21 learnable FFT filters — reads log-FFT only (Channel 1)
        # Purpose: Specialize exclusively in global frequency ring detection
        self.branch_c = nn.Conv2d(1, 21, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_c = nn.BatchNorm2d(21)

        self.relu = nn.ReLU(inplace=True)

        # Layer 2: Receives 11+53+21 = 85 merged channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(85, 16, kernel_size=3, stride=1, padding=1, bias=False),
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
        for name, m in self.named_modules():
            # Skip branch_a (SRM weights set manually above).
            # branch_b / branch_c keep PyTorch default kaiming_uniform_ — intentional.
            if any(x in name for x in ['branch_a', 'branch_b', 'branch_c']):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 2, H, W)
        spatial = x[:, 0:1, :, :]
        freq = x[:, 1:2, :, :]

        out_a = self.bn_a(self.branch_a(spatial))
        out_b = self.bn_b(self.branch_b(spatial))
        out_c = torch.abs(self.bn_c(self.branch_c(freq)))

        # Merge isolated domain features into 85-channel map
        x = self.relu(torch.cat([out_a, out_b, out_c], dim=1))

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