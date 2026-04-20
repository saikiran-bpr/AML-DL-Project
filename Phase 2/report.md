# Phase 2 Report: Autoencoder-Based IDS (NSL-KDD)

## 1) Model Explanation

### 1.1 Autoencoder Architecture
Phase 2 uses a deep autoencoder (`AutoencoderSkip`) with:
- Encoder: `input -> 128 -> 64 -> 32` (bottleneck)
- Decoder: `32 -> 64 -> 128 -> input`
- Skip connections between encoder and decoder blocks to preserve useful feature detail.

Dropout and batch normalization are applied to improve robustness and stabilize optimization on tabular network data.

### 1.2 Training on Normal Data Only
The model is trained only on normal samples from NSL-KDD train split.  
This follows one-class anomaly detection assumptions for IDS deployment where labeled attack data may be incomplete.

### 1.3 Reconstruction Error
For each sample, anomaly score is:
- mean squared reconstruction error across features.

Higher error implies lower conformity to learned normal traffic manifold and therefore higher anomaly likelihood.

## 2) Research Paper Integration

Paper used: **"Autoencoder-Based Anomaly Detection in Network Traffic"**.

### 2.1 What Is Adopted
- Reconstruction-based anomaly detection pipeline.
- One-class training perspective focused on normal behavior modeling.
- Threshold-based anomaly decision from reconstruction score distribution.

### 2.2 Paper Adaptation (What Is Modified)
- Added skip connections for better feature retention in deep tabular reconstruction.
- Used weighted MSE to reduce dominance of high-variance dimensions.
- Added dropout and batch normalization for regularization/stability.
- Added curriculum schedule over normal samples for smoother optimization.

### 2.3 Why Changes Were Needed for NSL-KDD
NSL-KDD is heterogeneous tabular traffic data with mixed feature scales and category structure.  
The above modifications improve reconstruction stability and anomaly separability under this dataset's characteristics.

## 3) Methodology

## 3.1 Data Split Strategy
- Train set: 80% of normal records only (from NSL-KDD train file).
- Validation set: remaining normal records + sampled attacks from NSL-KDD train file.
- Test set: NSL-KDD official test file, used only for final reporting.

## 3.2 Leakage Controls
- Categorical encoders are fitted on **training data only**.
- Unseen test categories are mapped to `-1` (no test-informed fitting).
- `StandardScaler` is fitted on train-normal data only.
- Threshold is tuned on validation set only.
- Test set is never used for threshold/hyperparameter tuning.

## 3.3 Threshold Selection
- Compute validation reconstruction errors.
- Search threshold over validation score range.
- Select threshold maximizing validation F1 (`find_threshold`).
- Apply the selected threshold unchanged to test scores.

## 4) Results

Primary metrics reported:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Artifacts:
- AE metrics: `results/ae_metrics.csv`
- Confusion matrix figure from AE evaluation notebook.
- ROC curve figure from comparison notebook.

## 5) Comparison With Phase 1

Comparison file:
- `results/full_comparison.csv`

Phase-1 baselines are loaded directly from:
- `../Phase 1/results/baseline_results.csv`

Compared models:
- Z-score (Phase 1)
- Isolation Forest (Phase 1)
- Autoencoder and other Phase-2 models (where executed)

This ensures reproducible and consistent Phase-1 vs Phase-2 evaluation without hardcoded baseline values.

## 6) Discussion

- Autoencoder can underperform or outperform baselines depending on threshold sensitivity and reconstruction overlap between normal and attack classes.
- ROC-AUC should be examined with F1 and confusion matrix to avoid relying on a single metric.
- Precision/recall trade-off is critical in IDS where false negatives and false positives have different operational costs.

### Limitations
- Dependence on threshold quality and validation distribution.
- Possible under-detection of subtle attacks that reconstruct similarly to normal traffic.
- Results can vary with architecture depth, regularization, and score calibration.

## 7) Final Validation Checklist

- [x] No LabelEncoder fit on test data.
- [x] Scaling fitted on training-normal data only.
- [x] Train-only normal data for AE fitting.
- [x] Validation-based threshold tuning.
- [x] Test-only final evaluation.
- [x] Baseline comparison loaded from Phase-1 CSV, not hardcoded.
- [x] Modular code structure preserved (`utils/data_loader.py`, `utils/models.py`, `utils/evaluation.py`).
- [x] Documentation completed (`README.md`, `report.md`).

