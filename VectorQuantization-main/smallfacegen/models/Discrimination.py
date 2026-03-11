import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator — classifies overlapping image patches as real/fake.
    Returns a spatial map of real/fake scores (not a single scalar),
    which provides denser gradient signal and better local texture supervision.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()

        # First layer: no normalization
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = base_channels
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 2 if i < n_layers - 1 else 1   # last hidden layer: stride=1
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm2d(nf, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Output layer: single-channel score map (no sigmoid — use with BCEWithLogitsLoss or hinge)
        layers += [
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  — real or reconstructed image
        Returns:
            patch_scores: (B, 1, H', W')  — raw logits per patch
        """
        return self.model(x)


# ---------- Loss helpers ----------

def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss_real = torch.mean(torch.relu(1.0 - real_logits))
    loss_fake = torch.mean(torch.relu(1.0 + fake_logits))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator (fool the discriminator)."""
    return -torch.mean(fake_logits)


def adopt_weight(weight: float, current_step: int, disc_start: int) -> float:
    """Return 0 before disc_start so GAN loss is disabled during warm-up."""
    return weight if current_step >= disc_start else 0.0


# ---------- Quick smoke test ----------
if __name__ == "__main__":
    disc = PatchGANDiscriminator(in_channels=3, base_channels=64, n_layers=3)
    x = torch.randn(2, 3, 128, 128)
    out = disc(x)
    print("Discriminator output shape:", out.shape)   # (2, 1, H', W')