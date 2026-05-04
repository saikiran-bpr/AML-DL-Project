"""Evaluation, scoring, and visualization utilities for Phase 3."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

from .config import ATTACK_CLASSES


plt.rcParams.update({
    'figure.figsize': (12, 6), 'axes.titlesize': 14,
    'axes.labelsize': 12, 'font.size': 11, 'figure.dpi': 100
})
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Threshold tuning + binary metrics.
# ---------------------------------------------------------------------------
def find_threshold(scores, y_true, n_points=200):
    """Pick the threshold that maximises validation F1."""
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99)
    thresholds = np.linspace(lo, hi, n_points)
    best_f1, best_t = 0.0, lo
    for t in thresholds:
        preds = (scores > t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def binary_metrics(y_true, y_pred, scores, name):
    """Print classification report and return a row dict for the comparison table."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, scores) if scores is not None else float('nan')
    print(f"\n{'=' * 60}\n{name} — TEST RESULTS\n{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    if not np.isnan(auc):
        print(f"ROC-AUC: {auc:.4f}")
    return {'model': name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'auc': auc}


# ---------------------------------------------------------------------------
# Multi-class metrics for the Cascaded AE+XGBoost classifier.
# ---------------------------------------------------------------------------
def multiclass_metrics(y_true, y_pred, name):
    """Macro-averaged metrics + per-class report for {Normal, DoS, Probe, R2L, U2R}."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n{'=' * 60}\n{name} — MULTI-CLASS TEST RESULTS\n{'=' * 60}")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(ATTACK_CLASSES))),
        target_names=ATTACK_CLASSES,
        zero_division=0,
    ))
    return {'model': name, 'accuracy': acc, 'macro_precision': prec,
            'macro_recall': rec, 'macro_f1': f1}


# ---------------------------------------------------------------------------
# Plotting helpers.
# ---------------------------------------------------------------------------
def plot_cm(y_true, y_pred, title, path, labels=('Normal', 'Anomaly'), cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_multiclass_cm(y_true, y_pred, title, path):
    plot_cm(y_true, y_pred, title, path, labels=ATTACK_CLASSES, cmap='Purples')


def plot_scores(scores, y_true, threshold, title, path):
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
    """roc_data: list of (name, y_true, scores, color)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, y_true, scores, color in roc_data:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_metric_bars(df, metric, title, path, color='#3498db'):
    """Horizontal bar chart for any single metric column from a comparison DataFrame."""
    df_sorted = df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df_sorted))))
    bars = ax.barh(df_sorted['model'], df_sorted[metric], color=color)
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)
    ax.set_xlabel(metric)
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, max(1.0, df_sorted[metric].max() * 1.1))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
