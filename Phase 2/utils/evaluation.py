"""Evaluation, scoring, and visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


plt.rcParams.update({
    'figure.figsize': (12, 6), 'axes.titlesize': 14,
    'axes.labelsize': 12, 'font.size': 11, 'figure.dpi': 100
})
sns.set_style("whitegrid")


def compute_recon_error(model, X, device, batch_size=512):
    """Compute per-sample reconstruction error (MSE)."""
    model.eval()
    errors = []
    loader = DataLoader(TensorDataset(torch.FloatTensor(X)),
                        batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon = model(x)[0]
            mse = torch.mean((x - recon) ** 2, dim=1)
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


def compute_vae_score(model, X, device, n_samples=50, batch_size=512):
    """Compute VAE reconstruction probability via Monte Carlo sampling."""
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(torch.FloatTensor(X)),
                        batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            mu, logvar = model.encode(x)
            sample_errors = []
            for _ in range(n_samples):
                z = model.reparameterize(mu, logvar)
                recon = model.decode(z)
                e = torch.mean((x - recon) ** 2, dim=1)
                sample_errors.append(e)
            avg = torch.stack(sample_errors).mean(dim=0)
            scores.append(avg.cpu().numpy())
    return np.concatenate(scores)


def find_threshold(scores, y_true, n_points=200):
    """Find optimal threshold by maximizing F1 score on validation set."""
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99)
    thresholds = np.linspace(lo, hi, n_points)
    best_f1, best_t = 0, lo
    for t in thresholds:
        preds = (scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def eval_model(y_true, y_pred, scores, name):
    """Evaluate model and print classification report."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, scores)
    print(f"\n{'=' * 60}")
    print(f"{name} — TEST RESULTS")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    print(f"ROC-AUC: {auc:.4f}")
    return {'model': name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'auc': auc}


def plot_cm(y_true, y_pred, title, path, cmap='Blues'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scores(scores, y_true, threshold, title, path):
    """Plot anomaly score distribution with threshold."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores[y_true == 0], bins=50, alpha=0.6, color='#2ecc71',
            label='Normal', density=True)
    ax.hist(scores[y_true == 1], bins=50, alpha=0.6, color='#e74c3c',
            label='Anomaly', density=True)
    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold={threshold:.4f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(roc_data, title, path):
    """Plot multiple ROC curves.

    Args:
        roc_data: list of (name, y_true, scores, color) tuples
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, y_true, scores, color in roc_data:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
