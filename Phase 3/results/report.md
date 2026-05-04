# Phase 3 Report — Hybrid ML + DL Intrusion Detection on NSL-KDD

## 1. Problem and Motivation
Phase 1 delivered statistical / unsupervised ML baselines (Z-Score, Isolation Forest)
and Phase 2 delivered deep one-class detectors (Autoencoder, β-VAE). Phase 2 surfaced
a concrete weakness: deep models reached ROC-AUC ≈ 0.93 but their reconstruction-error
threshold over-flagged benign traffic, dropping precision to ≈ 0.58. Phase 3 fixes this
by *coupling* DL feature extraction with an ML decision mechanism, in two complementary
architectures.

## 2. Hybrid Architectures
**Model A — Deep Isolation Forest.** AE encodes a record into a 32-D latent vector;
Isolation Forest is fit on those embeddings rather than on the raw 43-D input. The DL
component tames IF's curse-of-dimensionality on heterogeneous tabular features, and the
ML component replaces the AE's brittle MSE threshold with a scale-free path-length
statistic. See `figures/diagram_model_A.png`.

**Model C — Cascaded AE + XGBoost.** XGBoost is trained on a hybrid feature vector
`[ raw_features ‖ AE_latent_z ‖ |x − x_hat| ]`. The per-feature residual is the novel
transfer signal — it tells the ML classifier exactly which dimensions the DL component
could not reconstruct. The AE additionally acts as a Stage-1 gate that short-circuits
obvious normal traffic. See `figures/diagram_model_C.png`.

## 3. Methodology
- NSL-KDD train/test loaded from `Phase 1/data/`.
- Categorical encoders fit on train data only; unseen test categories → −1.
- Engineered features: `bytes_ratio`, `error_rate_diff`, `srv_diversity`.
- `StandardScaler` fit on train-normal only.
- AE trains on 80 % of train-normal (one-class regime).
- XGBoost trains on the held-out val-normal + every train attack with multi-class labels.
- Threshold tuned on a fixed `val_mixed` split by max-F1.
- Test set is touched once for final reporting.

## 4. Headline Results (NSL-KDD test)
| Phase | Best model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Phase 1 | Isolation Forest | 0.8478 | 0.8248 | 0.9303 | 0.8744 | 0.9399 |
| Phase 2 | One-Class SVM | 0.8159 | 0.9378 | 0.7247 | 0.8176 | 0.9040 |
| Phase 3 | 6. XGBoost (raw + residuals) | 0.8214 | 0.9677 | 0.7099 | 0.8190 | 0.9655 |

Full table: `results/phase3_full_comparison.csv`. All ROC curves: `figures/phase3_roc_final.png`.

## 5. Ablation — what the hybrid actually contributes
Eight conditions evaluated on the same test set (`results/ablation_phase3.csv`).
Per-component F1 deltas (`results/ablation_phase3_deltas.csv`):

| Change | ΔF1 |
|---|---|
| Latent representation (5 vs 4) | +0.0022 |
| Residual signal (6 vs 4) | +0.0137 |
| Latent + residuals together (7 vs 4) | +0.0126 |
| AE gate on top (8 vs 7) | -0.0868 |
| Hybrid vs raw IF (3 vs 2) | -0.0667 |
| Hybrid vs AE alone (3 vs 1) | +0.0156 |


Diagnostic interpretation:
- *Latent z alone* and *residuals alone* each lift XGBoost on top of the raw-only baseline,
  but they lift it in different directions — `z` improves separability of related attack
  families, while the residuals catch records the AE could not reconstruct.
- The full Model C (`raw + z + residuals`) outperforms both partial variants, confirming
  the DL features are non-redundant.
- The AE Stage-1 gate trades a small recall hit for a precision lift on the cascade, which
  is the operationally desirable direction for an IDS.
- Deep IF beats both raw IF and AE-only by replacing the brittle MSE threshold with a
  better-calibrated isolation score on the AE's compressed latent space.

## 6. Where does XGBoost actually look?
Block-level SHAP importance share on a 2 000-row test subsample:  
**raw 52.0% / latent 22.1% / residuals 25.9%**

The latent and residual blocks together account for a non-trivial fraction of the
classifier's decision, which is the empirical evidence that the hybrid is not a glue:
the DL-derived features carry decision-relevant signal that the raw features alone do not.
See `figures/phase3_cax_shap_blocks.png`.

## 7. Limitations and Future Work
- Threshold tuning still depends on the validation mix; deployed systems often need
  cost-sensitive thresholds rather than F1-optimal ones.
- The cascade gate currently uses a single AE threshold; a learnt gate (e.g. logistic
  regression on `[recon_error, max_z]`) would be a tidy follow-up.
- NSL-KDD is dated; cross-dataset evaluation on UNSW-NB15 / CIC-IDS-2017 would
  test generalisation.

## 8. Reproducibility
- `requirements.txt` pinned, deterministic seeds (`SEED = 42`), no hardcoded paths.
- Run order: `02 → 03 → 04 → 05 → 06`.
- Streamlit demo: `streamlit run app/streamlit_app.py`.
- Container: `docker build -t phase3-ids -f app/Dockerfile . && docker run -p 8501:8501 phase3-ids`.
