"""Hybrid ML+DL architectures for Phase 3.

Two models are exposed:

* DeepIsolationForest (Model A) — autoencoder feature extractor + Isolation
  Forest decision boundary on the latent space.
* CascadedAE_XGB (Model C) — autoencoder anomaly gate followed by an XGBoost
  multi-class classifier whose input concatenates raw features, the AE latent
  vector, and the per-feature reconstruction residuals.

The autoencoder definition is duplicated here (rather than imported from
Phase 2) so Phase 3 is fully self-contained for grading reproducibility.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "xgboost is required for Phase 3. Install via `pip install xgboost`."
    ) from e


# ---------------------------------------------------------------------------
# Autoencoder building block (matches Phase 2's AutoencoderSkip architecture).
# ---------------------------------------------------------------------------
class AutoencoderSkip(nn.Module):
    """Deep autoencoder with U-net style skip connections for tabular data.

    Architecture: Input(dim) -> 128 -> 64 -> 32(z) -> 64+64 -> 128+128 -> dim.
    Skip connections preserve low-level feature structure that pure compression
    would otherwise discard, which matters for heterogeneous NSL-KDD columns.
    """
    def __init__(self, dim: int = 43, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc1 = nn.Sequential(
            nn.Linear(dim, 128), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.enc2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.enc3 = nn.Sequential(
            nn.Linear(64, latent_dim), nn.BatchNorm1d(latent_dim), nn.LeakyReLU(0.2))

        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64),
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
        d1 = torch.cat([d1, h2], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, h1], dim=1)
        return self.dec3(d2)

    def forward(self, x):
        z, h1, h2 = self.encode(x)
        return self.decode(z, h1, h2), z


def train_autoencoder(model, X_train, device, epochs=60, batch_size=256, lr=1e-3,
                      val_X=None, verbose=True):
    """Train AE on normal data with weighted MSE; return per-epoch loss curves."""
    weights = torch.FloatTensor(1.0 / (X_train.var(axis=0) + 1e-3)).to(device)
    weights = weights / weights.mean()

    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_X is not None:
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_X)),
                                batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_hist, val_hist = [], []
    model.to(device)
    for ep in range(epochs):
        model.train()
        running = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = (((recon - xb) ** 2) * weights).mean()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        train_hist.append(running / len(loader.dataset))

        if val_loader is not None:
            model.eval()
            v_running = 0.0
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    recon, _ = model(xb)
                    v_running += (((recon - xb) ** 2) * weights).mean().item() * xb.size(0)
            val_hist.append(v_running / len(val_loader.dataset))

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            tail = f" | val {val_hist[-1]:.5f}" if val_hist else ""
            print(f"  epoch {ep+1:>3}/{epochs} | train {train_hist[-1]:.5f}{tail}")

    return train_hist, val_hist


@torch.no_grad()
def encode_dataset(model, X, device, batch_size=512):
    """Run the encoder over X and return (latent_z, reconstructed_x)."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X)),
                        batch_size=batch_size, shuffle=False)
    zs, recons = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        recon, z = model(xb)
        zs.append(z.cpu().numpy())
        recons.append(recon.cpu().numpy())
    return np.concatenate(zs), np.concatenate(recons)


def recon_error(X, X_recon):
    """Per-sample mean squared reconstruction error (anomaly score)."""
    return np.mean((X - X_recon) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Model A — Deep Isolation Forest.
# ---------------------------------------------------------------------------
class DeepIsolationForest:
    """AE encoder + Isolation Forest on the latent space.

    The AE learns a non-linear projection of normal traffic; the IF then carves
    a robust isolation-based boundary in that compact latent space. This swaps
    the AE's brittle MSE threshold for IF's path-length statistic, which is
    scale-free and well-calibrated even when the latent distribution is not
    Gaussian.
    """
    def __init__(self, ae: AutoencoderSkip, device, n_estimators: int = 200,
                 contamination: float = 0.05, max_samples: str | int = 'auto'):
        self.ae = ae
        self.device = device
        self.iforest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=42,
            n_jobs=-1,
        )
        self.fitted = False

    def fit(self, X_train_normal):
        """Fit IF on the AE latent embeddings of training normals."""
        z, _ = encode_dataset(self.ae, X_train_normal, self.device)
        self.iforest.fit(z)
        self.fitted = True
        return self

    def score(self, X):
        """Higher = more anomalous. We negate IF's score_samples (where higher = more normal)."""
        assert self.fitted, "DeepIsolationForest not fitted."
        z, _ = encode_dataset(self.ae, X, self.device)
        return -self.iforest.score_samples(z)

    def predict(self, X, threshold):
        return (self.score(X) > threshold).astype(int)


# ---------------------------------------------------------------------------
# Model C — Cascaded AE + XGBoost.
# ---------------------------------------------------------------------------
class CascadedAE_XGB:
    """Two-stage IDS: AE detector gate followed by multi-class XGBoost.

    Feature vector for the classifier:
        concat( raw_features, AE_latent_z, per_feature_residuals )

    Per-feature residuals (|x - x_hat|) are the novel signal here — they tell
    XGBoost *which* dimensions the autoencoder failed to reconstruct, turning
    the AE into a learned feature extractor for the supervised classifier.
    """
    def __init__(self, ae: AutoencoderSkip, device,
                 use_raw: bool = True, use_latent: bool = True,
                 use_residuals: bool = True,
                 xgb_params: dict | None = None):
        self.ae = ae
        self.device = device
        self.use_raw = use_raw
        self.use_latent = use_latent
        self.use_residuals = use_residuals

        defaults = dict(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', num_class=5,
            tree_method='hist', n_jobs=-1, random_state=42,
            eval_metric='mlogloss',
        )
        if xgb_params:
            defaults.update(xgb_params)
        self.xgb = XGBClassifier(**defaults)
        self.detector_threshold = None
        self.fitted = False

    def build_features(self, X):
        """Return the concatenated feature matrix used by the XGBoost stage."""
        z, recon = encode_dataset(self.ae, X, self.device)
        residuals = np.abs(X - recon)
        parts = []
        if self.use_raw:
            parts.append(X)
        if self.use_latent:
            parts.append(z)
        if self.use_residuals:
            parts.append(residuals)
        if not parts:
            raise ValueError("At least one of use_raw/use_latent/use_residuals must be True.")
        return np.hstack(parts)

    def fit(self, X_clf_train, y_clf_train_multi, detector_threshold: float | None = None):
        """Train the supervised XGBoost stage on AE-derived features.

        detector_threshold (anomaly-score cut for Stage 1) is supplied from the
        AE-only validation tuning so the cascade gate is calibrated.
        """
        feats = self.build_features(X_clf_train)
        self.xgb.fit(feats, y_clf_train_multi)
        self.detector_threshold = detector_threshold
        self.fitted = True
        return self

    def predict_multi(self, X):
        """Multi-class prediction (no Stage-1 gating — useful for ablations)."""
        assert self.fitted
        return self.xgb.predict(self.build_features(X))

    def predict_proba_multi(self, X):
        return self.xgb.predict_proba(self.build_features(X))

    def predict_cascade(self, X, normal_class_idx: int = 0):
        """Full cascade: Stage 1 gates with the AE; Stage 2 classifies the rest.

        Records below the AE threshold are accepted as Normal without paying for
        the XGBoost call. Records above the threshold get a multi-class label;
        if the classifier returns Normal we trust the gate and override to the
        most likely *attack* class (gate says anomalous, so don't say normal).
        """
        if self.detector_threshold is None:
            return self.predict_multi(X)

        z, recon = encode_dataset(self.ae, X, self.device)
        recon_score = np.mean((X - recon) ** 2, axis=1)
        flagged = recon_score > self.detector_threshold

        out = np.full(len(X), normal_class_idx, dtype=int)
        if flagged.any():
            feats = self.build_features(X[flagged])
            proba = self.xgb.predict_proba(feats)
            preds = proba.argmax(axis=1)
            # Override Normal predictions on gated rows to the top attack class —
            # the gate already said this row is anomalous.
            mask_normal = preds == normal_class_idx
            if mask_normal.any():
                attack_proba = proba[mask_normal].copy()
                attack_proba[:, normal_class_idx] = -1.0
                preds[mask_normal] = attack_proba.argmax(axis=1)
            out[flagged] = preds
        return out

    def predict_binary_score(self, X):
        """Anomaly score for binary evaluation = 1 - P(Normal) from XGBoost."""
        proba = self.predict_proba_multi(X)
        return 1.0 - proba[:, 0]
