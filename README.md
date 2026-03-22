# Adaptive Cyber-Physical Security: Anomaly-Based Intrusion Detection

## Project Overview

This project develops anomaly detection models for network intrusion detection, targeting **zero-day** and unknown cyber threats in industrial networks, IoT systems, and enterprise infrastructure. Unlike signature-based systems, our approach learns a profile of *normal* network behavior and detects deviations — enabling detection of previously unseen attack patterns.

## Dataset: NSL-KDD

The [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) dataset (Tavallaee et al., 2009) is an improved version of the KDD Cup '99 benchmark:

- **Training set:** 125,973 records (67,343 normal + 58,630 attack)
- **Test set:** 22,544 records with unseen attack types
- **Features:** 41 network traffic attributes (TCP connection, content, and traffic statistics)
- **Attack categories:** DoS, Probe, R2L (Remote-to-Local), U2R (User-to-Root)
- **Key improvement:** No duplicate records, balanced difficulty levels

## Project Structure

```
AML-DL-Project/
├── baseline_model.ipynb     # Phase 1: Baseline ML anomaly detection
├── data/                    # NSL-KDD dataset (auto-downloaded)
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── results/                 # Generated figures and metrics
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── dimensionality_reduction.png
│   ├── zscore_threshold_tuning.png
│   ├── zscore_results.png
│   ├── if_contamination_tuning.png
│   ├── isolation_forest_results.png
│   ├── roc_comparison.png
│   └── baseline_results.csv
└── README.md
```

## Baseline Models (Phase 1)

| Model | Approach | Training Strategy |
|-------|----------|-------------------|
| Z-Score Statistical Thresholding | Parametric — flags samples with extreme feature deviations | Trained on normal data only |
| Isolation Forest | Tree-based — isolates anomalies via random partitioning | Trained on normal data only |

Both models follow a **semi-supervised** paradigm: trained exclusively on benign traffic to detect unknown anomalies at inference time.

## How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Execution

```bash
# Launch Jupyter and open the notebook
jupyter notebook baseline_model.ipynb

# Or run non-interactively
jupyter nbconvert --to notebook --execute baseline_model.ipynb --output executed_baseline.ipynb
```

The notebook will:
1. Download the NSL-KDD dataset automatically (first run only)
2. Perform EDA and preprocessing
3. Train and evaluate both baseline models
4. Save all figures and metrics to `results/`

## References

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1–58.
2. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *ICDM*, 413–422.
3. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE CISDA*.
