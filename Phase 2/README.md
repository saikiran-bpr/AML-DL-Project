# Phase 2: Deep Learning IDS on NSL-KDD

## Problem Statement
Phase 2 extends the IDS project from statistical/ML baselines (Phase 1) to deep-learning anomaly detection.  
Goal: learn normal NSL-KDD traffic behavior and detect attacks as anomalies using reconstruction error.

## Phase-2 Approach (Autoencoder-Based Anomaly Detection)
- **Model**: deep autoencoder with encoder, bottleneck, decoder, and skip connections.
- **Training regime**: semi-supervised one-class learning (train on normal traffic only).
- **Scoring**: anomaly score is per-sample reconstruction MSE.
- **Decision rule**: classify as anomaly if reconstruction error is above a validation-tuned threshold.

### Research Paper Basis
Primary inspiration: **"Autoencoder-Based Anomaly Detection in Network Traffic"**.

This implementation adopts the paper's core reconstruction-based IDS workflow and adapts it to the NSL-KDD tabular setting used in this project.

## Data Pipeline and Leakage Prevention
- Load NSL-KDD train/test files from `Phase 1/data`.
- Encode categorical columns (`protocol_type`, `service`, `flag`) with a **train-only fitted mapping**.
- Map unseen test categories to `-1` (safe handling without fitting on test values).
- Engineer domain features (`bytes_ratio`, `error_rate_diff`, `srv_diversity`).
- Split train data:
  - **Train**: 80% of normal records only (model fitting)
  - **Validation**: remaining normal + sampled attacks from train split (threshold tuning)
  - **Test**: held-out NSL-KDD test set (final evaluation only)
- Fit `StandardScaler` on train-normal data only, then transform validation and test sets.

## Why Reconstruction Error Works
The autoencoder is optimized to reconstruct normal traffic patterns.  
Attack/anomalous samples are less consistent with learned normal manifold, so they produce larger reconstruction error.

## Improvements Beyond Base Paper
- **Skip connections**: preserve informative low-level feature structure in tabular traffic data.
- **Weighted MSE loss**: balances dimensions with very different variances.
- **Dropout + BatchNorm**: improves generalization and training stability.
- **Curriculum training**: starts from easier normal samples and gradually includes harder ones.

## Repository Structure (Phase 2)
- `03_data_preprocessing.ipynb`: preprocessing and split generation.
- `05_autoencoder_training.ipynb`: AE training and AE evaluation.
- `07_comparison_ablation.ipynb`: Phase 1 vs Phase 2 comparison and ablations.
- `utils/data_loader.py`: loading, encoding, split creation.
- `utils/models.py`: AE/VAE architectures and losses.
- `utils/evaluation.py`: scoring, thresholding, metrics, plots.
- `results/`: saved metrics/plots (`ae_metrics.csv`, `full_comparison.csv`, ROC/confusion images, etc.).

## How To Run
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn torch matplotlib seaborn jupyter
   ```
2. Run preprocessing notebook:
   - `Phase 2/03_data_preprocessing.ipynb`
   - Produces `results/processed_data.npz` and `results/scaler.pkl`.
3. Run autoencoder training notebook:
   - `Phase 2/05_autoencoder_training.ipynb`
   - Produces AE metrics and plots (confusion matrix / score distribution).
4. Run comparison notebook:
   - `Phase 2/07_comparison_ablation.ipynb`
   - Loads Phase 1 metrics from `Phase 1/results/baseline_results.csv` (no hardcoded baselines).
   - Produces `results/full_comparison.csv` and comparison figures.

## Results Summary (Phase 1 vs Phase 2)
Phase-1 and Phase-2 metrics are consolidated in:
- `Phase 2/results/full_comparison.csv`

The comparison includes:
- Z-score (Phase 1)
- Isolation Forest (Phase 1)
- Phase 2 models (One-Class SVM, Autoencoder, VAE where available)

Interpretation guidance:
- Compare F1 and ROC-AUC primarily for anomaly-detection quality.
- Use confusion matrices to inspect false-positive/false-negative behavior.

