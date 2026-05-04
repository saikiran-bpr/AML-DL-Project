# Phase 3: Hybrid ML + DL Intrusion Detection on NSL-KDD

## Problem Statement
Phase 3 unifies the classical ML baselines from Phase 1 (Z-Score, Isolation Forest)
and the deep one-class detectors from Phase 2 (Autoencoder, VAE) into two
**hybrid architectures** that explicitly couple the two paradigms.

The motivation comes directly from a Phase-2 weakness: the deep autoencoder
achieved ROC-AUC = 0.93 but precision was only 0.58 on the held-out NSL-KDD test
set. Recall saturated at 1.0, indicating the reconstruction-error threshold was
too permissive. Phase 3 replaces this brittle threshold with an ML decision
mechanism operating on the AE's latent space, and it adds a supervised
attack-family classifier for triage.

## Hybrid Architectures

### Model A — Deep Isolation Forest (DIF)
The Phase-2 autoencoder encodes a record into a 32-dimensional latent vector
`z`. An Isolation Forest is then trained on `z` (not on the raw 43-D input).
The final anomaly score is the IF isolation score in latent space.

**Why this is a hybrid, not glue:**
- The AE solves IF's curse-of-dimensionality on heterogeneous tabular features.
- IF replaces the AE's arbitrary MSE threshold with a principled isolation-based
  decision boundary, fixing the over-flagging behaviour observed in Phase 2.

### Model C — Cascaded AE + XGBoost (CAX)
Two-stage IDS pipeline:
- **Stage 1 (DL, semi-supervised):** AE flags anomaly via reconstruction error.
- **Stage 2 (ML, supervised):** XGBoost takes the concatenated feature vector
  `[raw_features ‖ AE_latent_z ‖ per_feature_residuals]` and classifies into
  `{Normal, DoS, Probe, R2L, U2R}`.

The per-feature residuals are the novel signal — they tell XGBoost *which*
dimensions the AE failed to reconstruct, transferring information from the
DL component into the ML component.

## Repository Structure
```
Phase 3/
├── 01_introduction_hybrid.ipynb     # Motivation + literature
├── 02_data_preparation.ipynb        # Splits + preprocessing (binary + multi-class)
├── 03_deep_isolation_forest.ipynb   # Model A
├── 04_cascaded_ae_xgboost.ipynb     # Model C
├── 05_ablation_study.ipynb          # ML-only vs DL-only vs Hybrid table
├── 06_final_comparison.ipynb        # Phase 1 vs Phase 2 vs Phase 3 + report
├── utils/
│   ├── config.py                    # Paths, seeds, device, NSL-KDD schema
│   ├── data_loader.py               # NSL-KDD loading + leakage-safe splits
│   ├── hybrid_models.py             # DeepIsolationForest, CascadedAE_XGB, AE
│   └── evaluation.py                # Binary + multi-class metrics, plotting
├── app/
│   ├── streamlit_app.py             # Live demo UI (Extra Mile)
│   └── Dockerfile                   # Reproducible container (Extra Mile)
├── models/                          # Saved AE, IF, XGBoost, scaler
├── results/                         # CSVs and metric tables
├── figures/                         # PNG figures for the report
├── references/                      # User-provided references
├── requirements.txt
└── README.md
```

## Data Pipeline (Leakage-Safe)
- NSL-KDD train/test loaded from `../Phase 1/data/`.
- Categorical encoders fit on training data only; unseen test categories → -1.
- Engineered features: `bytes_ratio`, `error_rate_diff`, `srv_diversity`.
- `StandardScaler` fit on train-normal only.
- AE trains on 80 % of train-normal (Phase-2 split, preserved here).
- XGBoost trains on the *remaining* train rows (held-out normal + all attacks)
  with attack-family labels — never sees AE's training samples.
- Threshold and hyperparameters tuned on the validation split only.
- Final NSL-KDD test set used exactly once for reporting.

## How to Run
```bash
pip install -r requirements.txt

# Run notebooks in order
jupyter nbconvert --to notebook --execute 02_data_preparation.ipynb --output 02_data_preparation.ipynb
jupyter nbconvert --to notebook --execute 03_deep_isolation_forest.ipynb --output 03_deep_isolation_forest.ipynb
jupyter nbconvert --to notebook --execute 04_cascaded_ae_xgboost.ipynb --output 04_cascaded_ae_xgboost.ipynb
jupyter nbconvert --to notebook --execute 05_ablation_study.ipynb --output 05_ablation_study.ipynb
jupyter nbconvert --to notebook --execute 06_final_comparison.ipynb --output 06_final_comparison.ipynb
```

### Streamlit demo (Extra Mile)
```bash
streamlit run app/streamlit_app.py
```

### Docker (Extra Mile)
```bash
docker build -t phase3-ids -f app/Dockerfile .
docker run -p 8501:8501 phase3-ids
```

## Results Summary
Final consolidated metrics live in `results/phase3_full_comparison.csv` and
include every model from Phase 1, Phase 2, and Phase 3 evaluated on the same
held-out NSL-KDD test set. See `06_final_comparison.ipynb` for figures and
`results/report.md` for the written analysis.
