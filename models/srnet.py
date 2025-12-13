import torch
import torch.nn as nn

class SRNet(nn.Module):
    """
    Architecture:
    - Layer 1-2: Feature extraction (no pooling, no shortcuts)
    - Layer 3-7: Residual blocks (no pooling)
    - Layer 8-11: Residual blocks with pooling
    - Layer 12: Global pooling and classification
    """

    def __init__(self, num_classes=2):
        super(SRNet, self).__init__()

        # Layer 1: Initial convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
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

        # Global pooling
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride=1, pool=False):
        """Create a residual block"""
        layers = []

        # Main path
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        # Pooling layer if specified
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        return ResidualBlock(nn.Sequential(*layers), in_channels, out_channels, pool)

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input should be grayscale image (batch_size, 1, H, W)

        # Layers 1-2: Feature extraction
        x = self.layer1(x)
        x = self.layer2(x)

        # Layers 3-7: Residual blocks without pooling
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # Layers 8-11: Residual blocks with pooling
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        # Global pooling
        x = self.global_pooling(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block with optional pooling"""

    def __init__(self, main_path, in_channels, out_channels, pool=False):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.pool = pool

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # Pooling for shortcut if needed
        if pool:
            self.shortcut_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # Main path
        out = self.main_path(x)

        # Shortcut path
        identity = self.shortcut(identity)
        if self.pool:
            identity = self.shortcut_pool(identity)

        # Add and activate
        out = out + identity
        out = self.relu(out)

        return out