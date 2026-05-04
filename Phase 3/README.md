# Phase 3 — Hybrid ML + DL Intrusion Detection on NSL-KDD

**Authors:** Prerak Arya (230039), Sai Kiran Bompelliwar (230046) — Rishihood University
**Dataset:** NSL-KDD (Tavallaee et al., 2009)
**Runtime:** PyTorch 2.x (MPS / CUDA / CPU) · scikit-learn 1.x · XGBoost 3.x

This repository extends Phase 1 (statistical / classical ML baselines) and Phase 2
(deep one-class detectors) with two **hybrid architectures** that explicitly
couple deep representation learning with classical machine learning. Each
notebook is self-contained, every artefact is regenerable, and the full
pipeline runs end-to-end via a single shell script.

---

## 1. Why hybrids

Phase 2 surfaced a calibration weakness: the deep autoencoder reached
ROC-AUC ≈ 0.93 on the NSL-KDD test set but precision collapsed to 0.58 because
the reconstruction-error threshold over-flagged benign traffic. Phase 3
replaces that brittle threshold with a machine-learning decision rule and
adds a supervised attack-family classifier on top.

| | Strength | Weakness on NSL-KDD |
|---|---|---|
| Classical ML (IF, OCSVM) | Calibrated, interpretable decision rules | Curse of dimensionality on heterogeneous tabular features |
| Deep AE (Phase 2) | Captures non-linear feature correlations | Threshold is brittle and uncalibrated |

Each Phase 3 hybrid pairs the two so that one half fixes the other's weakness.

---

## 2. The two hybrids

### Model A — Deep Isolation Forest (DIF)
```
x ∈ ℝ⁴³  →  AE encoder (DL)  →  z ∈ ℝ³²  →  Isolation Forest (ML)  →  anomaly score
```
The autoencoder gives Isolation Forest a dense, decorrelated latent. Isolation
Forest replaces the autoencoder's MSE threshold with a path-length statistic
that is scale-free and well-calibrated. Inspired by Xu et al., *Deep Isolation
Forest*, IEEE TKDE 2023.

### Model C — Cascaded AE + XGBoost (CAX)
```
                    ┌────► z          (32-D latent)
x ∈ ℝ⁴³ → AE (DL) ──┤
                    └────► |x − x̂|   (43-D residual)

concat(x, z, |x − x̂|) ∈ ℝ¹¹⁸  →  XGBoost (ML)  →  ŷ ∈ {Normal, DoS, Probe, R2L, U2R}
```
XGBoost consumes three views of every record: the raw features, the AE latent,
and the per-feature reconstruction residuals. The residual block is the
architectural innovation — it is a signal trees cannot derive from raw input on
their own. Inspired by Shone et al., IEEE TETCI 2018, with XGBoost
(Chen & Guestrin, KDD 2016) replacing Random Forest.

---

## 3. Repository layout

```
Phase 3/
├── 01_introduction_hybrid.ipynb     Motivation, both architectures, rubric mapping
├── 01b_literature_review.ipynb      Four-paper anchor review (2 per model)
├── 02_data_preparation.ipynb        Leakage-safe splits, multi-class labels
├── 03_deep_isolation_forest.ipynb   Model A — train AE, fit IF on latent
├── 04_cascaded_ae_xgboost.ipynb     Model C — train XGBoost on hybrid features + SHAP
├── 05_ablation_study.ipynb          Eight-condition ablation, per-component F1 deltas
├── 06_final_comparison.ipynb        Cross-phase table, ROC, diagrams, report.md
├── main.tex                         IEEE conference-style report (Overleaf-ready)
├── run_all.sh                       One-shot pipeline runner + artifact verifier
├── requirements.txt
├── utils/
│   ├── config.py                    Paths, seeds, device, NSL-KDD schema
│   ├── data_loader.py               NSL-KDD loading + leakage-safe splits
│   ├── hybrid_models.py             AE, DeepIsolationForest, CascadedAE_XGB
│   └── evaluation.py                Binary + multi-class metrics, plotting
├── app/
│   ├── streamlit_app.py             Live demo UI
│   └── Dockerfile                   Reproducible container
├── models/                          Trained AE, IF, XGBoost, scaler
├── results/                         CSVs, NPZ score archives, report.md
├── figures/                         Diagrams, ROC, ablation bars, SHAP, demo screenshot
└── references/                      Source PDFs of the four anchor papers
```

---

## 4. Quick start

```bash
cd "Phase 3"
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` executes notebooks 02–06 in order, then verifies that every
required artefact landed on disk. It exports `OMP_NUM_THREADS=1` first to avoid
a known macOS interaction (PyTorch and XGBoost both bundle `libomp.dylib`; the
second OpenMP runtime to load can silently kill the kernel).

To run only a single notebook by hand:
```bash
OMP_NUM_THREADS=1 jupyter nbconvert --to notebook --execute --inplace 04_cascaded_ae_xgboost.ipynb
```

---

## 5. Data pipeline (leakage-safe)

| Partition | Source | Used by |
|---|---|---|
| `X_train_normal` | 80 % of train normals | AE training (one-class) |
| `X_val_normal`   | 20 % of train normals | AE validation curve |
| `X_val_mixed`    | val normal + sampled attacks | Threshold tuning (max-F1) |
| `X_clf_train`    | val normal + every train attack (multi-class labels) | XGBoost training |
| `X_test`         | NSL-KDD `KDDTest+` | Final reporting only |

* Categorical encoders fit on training data only; unseen test categories → −1.
* Engineered features: `bytes_ratio`, `error_rate_diff`, `srv_diversity`.
* `StandardScaler` fit on AE-train normals only.
* AE training rows never overlap the XGBoost training pool, so the residuals
  XGBoost sees at fit time are realistic.
* The test set is touched exactly once, for final reporting.

---

## 6. Results

### Cross-phase headline (NSL-KDD test, 22 544 rows)

| Phase | Model | Acc | Prec | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| 1 | Isolation Forest          | 0.848 | 0.825 | 0.930 | **0.874** | 0.940 |
| 1 | Z-Score                   | 0.853 | 0.887 | 0.850 | 0.868 | 0.899 |
| 2 | One-Class SVM             | 0.816 | 0.938 | 0.725 | 0.818 | 0.904 |
| 2 | β-VAE                     | 0.582 | 0.577 | 1.000 | 0.732 | 0.941 |
| 2 | Autoencoder               | 0.582 | 0.576 | 1.000 | 0.731 | 0.932 |
| **3** | **Model C — raw + residuals** | **0.821** | **0.968** | 0.710 | **0.819** | **0.966** |
| **3** | **Model C — full**            | 0.820 | 0.967 | 0.709 | 0.818 | 0.964 |
| 3 | Deep IF (Model A)         | 0.758 | 0.926 | 0.626 | 0.747 | 0.927 |

Phase 3 leads on **AUC** and **precision** across the whole table. Phase 1
Isolation Forest still has the highest **F1** — the Phase 3 contribution is a
calibration shift (substantially higher precision, slightly lower F1) plus the
multi-class triage capability and per-component diagnostic evidence that the
classical baselines cannot offer.

Source: `results/phase3_full_comparison.csv`.

### Eight-condition ablation

The diagnostic teardown that proves where the gain comes from. All conditions
evaluated on the same test set with the same threshold-tuning protocol.

| # | Condition | F1 | AUC |
|---|---|---:|---:|
| 1 | AE only                       | 0.731 | 0.954 |
| 2 | Isolation Forest (raw)        | 0.813 | 0.937 |
| 3 | Deep IF (Model A)             | 0.747 | 0.927 |
| 4 | XGBoost (raw only)            | 0.805 | 0.960 |
| 5 | XGBoost (raw + latent z)      | 0.807 | 0.963 |
| 6 | XGBoost (raw + residuals)     | **0.819** | **0.966** |
| 7 | Model C (raw + z + residuals) | 0.818 | 0.964 |
| 8 | Model C + AE gate (cascade)   | 0.731 | 0.964 |

Per-component F1 deltas (`results/ablation_phase3_deltas.csv`):

| Change | ΔF1 |
|---|---:|
| Latent representation (5 vs 4)             | +0.002 |
| Residual signal (6 vs 4)                   | **+0.014** |
| Latent + residuals together (7 vs 4)       | +0.013 |
| AE gate on top (8 vs 7)                    | **−0.087** |
| Hybrid vs raw IF (3 vs 2)                  | −0.067 |
| Hybrid vs AE alone (3 vs 1)                | +0.016 |

**Honest findings reported as-is:**
* The **residual block carries the supervised gain.** Latent alone adds
  almost nothing (+0.002 F1).
* **Latent and residuals are largely redundant** for binary detection. The
  full hybrid (#7) is essentially tied with `raw + residuals` (#6).
* The **Stage-1 AE gate hurts on this run** (−0.087 F1) because the AE-only
  threshold is permissive (recall 1.0, precision 0.58). The ungated Model C
  is the recommended deployment configuration.
* **Phase-1 Isolation Forest still has the highest F1 overall.** The Phase-3
  story is calibration (AUC, precision) and triage (multi-class output), not
  an F1 win.

### Does XGBoost actually use the deep features?

SHAP block-share attribution on a 2 000-row test subsample:

| Block | Share of mean \|SHAP\| |
|---|---:|
| Raw features          | 52.0 % |
| AE latent (z)         | 22.1 % |
| AE residuals          | 25.9 % |

**48 % of XGBoost's decision attribution comes from AE-derived features.**
This is the empirical evidence that Model C is a real hybrid rather than glue.
See `figures/phase3_cax_shap_blocks.png`.

---

## 7. Architecture diagrams

Drawn live in notebook 06 with matplotlib (no external graphics dependency),
so they regenerate every run.

![Model A — Deep Isolation Forest](figures/diagram_model_A.png)

![Model C — Cascaded AE + XGBoost](figures/diagram_model_C.png)

---

## 8. Streamlit demo

Run live inference on any NSL-KDD test record and see both hybrids' decisions
side by side, plus the top reconstruction residuals (the DL → ML signal):

```bash
streamlit run app/streamlit_app.py
```

![Streamlit demo](figures/streamlit_demo.png)

* Pick a record by random sampling, by attack family, or by row index.
* Model A returns the latent-IF anomaly score and the validation-tuned
  threshold.
* Model C returns the predicted attack family with full per-class
  probabilities.
* The top-15 per-feature residuals are shown as the explanation for the
  flag — they are exactly what XGBoost consumes.

---

## 9. Docker

Run from the repository root (the build context must include both `Phase 1/data`
and `Phase 3`):

```bash
docker build -t phase3-ids -f "Phase 3/app/Dockerfile" .
docker run --rm -p 8501:8501 phase3-ids
```

The container regenerates missing artefacts (notebooks 02–04) on first launch,
then serves the Streamlit demo on port 8501.

---

## 10. IEEE report

A complete IEEE conference-style write-up is in [`main.tex`](main.tex). Compile
on Overleaf by uploading `main.tex` together with the `figures/` folder; the
default pdfLaTeX compiler is sufficient. All numbers in the tables are sourced
from the CSVs in `results/` so the report stays in sync with the pipeline.

---

## 11. Verified artefacts

`run_all.sh` checks that the following exist after a successful run:

| Artefact | Purpose |
|---|---|
| `results/phase3_full_comparison.csv` | Cross-phase comparison table |
| `results/ablation_phase3.csv`        | Eight-condition ablation |
| `results/ablation_phase3_deltas.csv` | Per-component F1 deltas |
| `results/modelC_scores.npz`          | Model C scores + SHAP block share |
| `figures/diagram_model_A.png`        | Model A architecture diagram |
| `figures/diagram_model_C.png`        | Model C architecture diagram |

Additional outputs (not on the verifier's hard-fail list, but always produced):
all per-condition confusion matrices, ROC curves, ablation bar charts, the
SHAP block plot, the latent-space PCA scatter, and `results/report.md`.

---

## 12. References

The four anchor papers driving the hybrid design (full review in
`01b_literature_review.ipynb`):

1. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM*.
2. Xu, H., Pang, G., Wang, Y., & Wang, Y. (2023). Deep Isolation Forest for
   Anomaly Detection. *IEEE TKDE*.
3. Shone, N., Ngoc, T. N., Phai, V. D., & Shi, Q. (2018). A Deep Learning
   Approach to Network Intrusion Detection. *IEEE TETCI* 2(1), 41–50.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
   *ACM KDD*.

Background / dataset references: Tavallaee et al. (2009) for NSL-KDD; Sakurada
& Yairi (2014) for autoencoder anomaly detection; Chandola, Banerjee & Kumar
(2009) for the anomaly-detection taxonomy; Lundberg & Lee (2017) for SHAP.
