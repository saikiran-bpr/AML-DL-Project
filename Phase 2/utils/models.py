"""Neural network architectures for Phase 2."""
import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE loss with per-feature inverse-variance weighting.

    Weighting stabilizes training on NSL-KDD tabular features where
    some dimensions have much higher dispersion than others.
    """
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, recon, target):
        mse = (recon - target) ** 2
        if self.weights is not None:
            mse = mse * self.weights
        return mse.mean()


class AutoencoderSkip(nn.Module):
    """Deep Autoencoder with skip connections (residual learning for tabular data).

    Architecture: Input(dim) -> 128 -> 64 -> 32(bottleneck) -> 64 -> 128 -> dim
    Skip connections: enc1<->dec3, enc2<->dec2
    """
    def __init__(self, dim=43):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(dim, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.enc2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.enc3 = nn.Sequential(
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2))

        self.dec1 = nn.Sequential(
            nn.Linear(32, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.dec2 = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.dec3 = nn.Linear(256, dim)

    def encode(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        z = self.enc3(h2)
        return z, h1, h2

    def decode(self, z, h1, h2):
        d1 = self.dec1(z)
        # Skip links preserve useful low-level structure in tabular traffic
        # features while still enforcing compression at the bottleneck.
        d1 = torch.cat([d1, h2], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, h1], dim=1)
        return self.dec3(d2)

    def forward(self, x):
        z, h1, h2 = self.encode(x)
        recon = self.decode(z, h1, h2)
        return recon, z


class VanillaAE(nn.Module):
    """Standard Autoencoder without skip connections (ablation baseline)."""
    def __init__(self, dim=43):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.2))
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(128, dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class VAE(nn.Module):
    """Variational Autoencoder with configurable latent dimension.

    Architecture: Input(dim) -> 128 -> 64 -> [mu, logvar](latent_dim) -> 64 -> 128 -> dim
    """
    def __init__(self, dim=43, latent_dim=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(dim, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.enc2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.dec2 = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.dec3 = nn.Linear(128, dim)

    def encode(self, x):
        h = self.enc2(self.enc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.dec3(self.dec2(self.dec1(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss_fn(recon_x, x, mu, logvar, beta=1.0):
    """Compute beta-VAE loss = reconstruction + beta * KL divergence."""
    recon = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl
