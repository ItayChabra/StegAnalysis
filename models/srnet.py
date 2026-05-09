import torch
import torch.nn as nn
import numpy as np


# ── 30-filter SRM kernel bank ─────────────────────────────────────────────────
# Fridrich & Kodovsky 2012 ("Rich Models for Steganalysis of Digital Images");
# canonical bank used by Yedroudj-Net (2018) and Zhu-Net (2019).
#
# All 30 filters padded to 5×5 so they fit one Conv2d. Frozen weights — pure
# linear high-pass operators that extract noise residuals from the cover.
# Adaptive stego (HUGO / WOW / S-UNIWARD) hides in image noise, so residual
# extraction at the input is essential — random conv filters cannot recover
# this signal in any reasonable training budget.

def _build_srm_kernels() -> np.ndarray:
    K = np.zeros((30, 5, 5), dtype=np.float32)

    # Class 1 — 1st-order linear diffs (4 directions)
    K[0, 2, 2:4] = [-1,  1]
    K[1, 2, 1:3] = [ 1, -1]
    K[2, 2:4, 2] = [-1,  1]
    K[3, 1:3, 2] = [ 1, -1]

    # Class 2 — 2nd-order linear diffs (h, v, diag, anti-diag)
    K[4, 2, 1:4]                    = [1, -2,  1]
    K[5, 1:4, 2]                    = [1, -2,  1]
    K[6, [1, 2, 3], [1, 2, 3]]      = [1, -2,  1]
    K[7, [1, 2, 3], [3, 2, 1]]      = [1, -2,  1]

    # Class 3 — 3rd-order linear diffs
    K[8,  2, 0:4]                       = [1, -3,  3, -1]
    K[9,  0:4, 2]                       = [1, -3,  3, -1]
    K[10, [0, 1, 2, 3], [0, 1, 2, 3]]   = [1, -3,  3, -1]
    K[11, [0, 1, 2, 3], [3, 2, 1, 0]]   = [1, -3,  3, -1]

    # Class 4 — EDGE3: 1st-order edge stencils, 4 rotations
    edge3 = np.array([[-1, 2, -1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    for i in range(4):
        K[12 + i, 1:4, 1:4] = np.rot90(edge3, i)

    # Class 5 — EDGE5: 2nd-order edge stencils, 4 rotations
    edge5 = np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float32)
    for i in range(4):
        K[16 + i, 1:4, 1:4] = np.rot90(edge5, i)

    # Class 6 — KV (3×3 5th-order isotropic) — the workhorse SRM filter
    K[20, 1:4, 1:4] = [[-1,  2, -1],
                       [ 2, -4,  2],
                       [-1,  2, -1]]

    # Class 7 — KB (5×5 5th-order isotropic) — wider receptive field
    K[21] = [[-1,  2, -2,  2, -1],
             [ 2, -6,  8, -6,  2],
             [-2,  8,-12,  8, -2],
             [ 2, -6,  8, -6,  2],
             [-1,  2, -2,  2, -1]]

    # Class 8 — asymmetric 1st-order ridge detectors at 4 corners
    K[22, 1:4, 1:4] = [[ 1, -1,  0], [0, 0, 0], [ 0,  0,  0]]
    K[23, 1:4, 1:4] = [[ 0, -1,  1], [0, 0, 0], [ 0,  0,  0]]
    K[24, 1:4, 1:4] = [[ 0,  0,  0], [0, 0, 0], [ 0, -1,  1]]
    K[25, 1:4, 1:4] = [[ 0,  0,  0], [0, 0, 0], [ 1, -1,  0]]

    # Class 9 — diagonal 2nd-order
    K[26, 1:4, 1:4] = [[-1,  0,  0], [ 0,  2,  0], [ 0,  0, -1]]
    K[27, 1:4, 1:4] = [[ 0,  0, -1], [ 0,  2,  0], [-1,  0,  0]]

    # Class 10 — bilateral edge detectors
    K[28, 1:4, 1:4] = [[-1,  2, -1], [ 0,  0,  0], [-1,  2, -1]]   # top + bottom
    K[29, 1:4, 1:4] = [[-1,  0, -1], [ 2,  0,  2], [-1,  0, -1]]   # left + right

    return K


# Truncation threshold for the TLU activation after the SRM conv.
# Most natural-image residuals fall within ±3; values beyond that are noise
# outliers that destabilise BatchNorm and the downstream conv stack.
_TLU_THRESHOLD = 3.0


class SRNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SRNet, self).__init__()

        # ── Triple-branch frontend (proper 30-filter SRM) ────────────────────

        # Branch A: 30 frozen SRM high-pass filters → TLU → BN
        # kernel_size=5 accommodates both 3×3 and 5×5 SRM stencils in one Conv2d
        # (3×3 stencils sit inside a zero-padded 5×5 frame).
        self.branch_a = nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2, bias=False)
        srm = _build_srm_kernels()                   # (30, 5, 5) float32
        with torch.no_grad():
            self.branch_a.weight.copy_(torch.from_numpy(srm).unsqueeze(1))  # → (30, 1, 5, 5)
        self.branch_a.requires_grad_(False)
        self.bn_a = nn.BatchNorm2d(30)

        # Branch B: 53 learnable spatial filters. Picks up artefacts that don't
        # fall on the SRM filter span (LSB-style direct pixel modifications,
        # DCT-block boundary patterns). Kept at 53 channels so the proven
        # learnable-spatial capacity isn't reduced — SRM is added on top.
        self.branch_b = nn.Conv2d(1, 53, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(53)

        # Branch C: 21 learnable FFT filters — reads log-FFT only (channel 1).
        self.branch_c = nn.Conv2d(1, 21, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_c = nn.BatchNorm2d(21)

        self.relu = nn.ReLU(inplace=True)

        # Layer 2 receives 30 + 53 + 21 = 104 merged channels.
        # Frozen SRM is additive: complements learnable filters with canonical
        # noise-residual extraction that random init cannot easily discover.
        self.layer2 = nn.Sequential(
            nn.Conv2d(104, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Layers 3–7: residual blocks without pooling
        self.layer3 = self._make_layer(16, 16, stride=1)
        self.layer4 = self._make_layer(16, 16, stride=1)
        self.layer5 = self._make_layer(16, 16, stride=1)
        self.layer6 = self._make_layer(16, 16, stride=1)
        self.layer7 = self._make_layer(16, 16, stride=1)

        # Layers 8–11: residual blocks with pooling
        self.layer8  = self._make_layer(16,  16,  stride=1, pool=True)
        self.layer9  = self._make_layer(16,  64,  stride=1, pool=True)
        self.layer10 = self._make_layer(64,  128, stride=1, pool=True)
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
        # x: (B, 2, H, W) — channel 0 = spatial, channel 1 = log-FFT
        spatial = x[:, 0:1, :, :]
        freq    = x[:, 1:2, :, :]

        # Branch A: SRM residuals → TLU (clamp to ±T) → BN
        out_a = self.branch_a(spatial)
        out_a = torch.clamp(out_a, -_TLU_THRESHOLD, _TLU_THRESHOLD)
        out_a = self.bn_a(out_a)

        # Branch B: learnable spatial conv → BN
        out_b = self.bn_b(self.branch_b(spatial))

        # Branch C: learnable FFT conv → BN → abs (FFT magnitude is non-negative)
        out_c = torch.abs(self.bn_c(self.branch_c(freq)))

        # Merge isolated domain features into 104-channel map
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