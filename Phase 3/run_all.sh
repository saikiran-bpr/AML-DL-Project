#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p results figures

# macOS workaround: PyTorch (loaded in nb03/04) and XGBoost (loaded in nb04/05)
# both bundle their own libomp.dylib, and the second OpenMP runtime to load can
# silently kill the kernel. Pinning to one OMP thread before Python starts
# avoids the collision. PyTorch on MPS is GPU-bound, so this has no measurable
# impact on AE training time.
export OMP_NUM_THREADS=1

NOTEBOOKS=(
  "02_data_preparation.ipynb"
  "03_deep_isolation_forest.ipynb"
  "04_cascaded_ae_xgboost.ipynb"
  "05_ablation_study.ipynb"
  "06_final_comparison.ipynb"
)

echo "Phase 3 pipeline started."
echo "Working directory: $(pwd)"

for notebook in "${NOTEBOOKS[@]}"; do
  echo
  echo "Running ${notebook}..."
  python3 -m nbconvert --to notebook --execute --inplace "$notebook"
  echo "Finished ${notebook}."
done

echo
echo "Verifying required Phase 3 artifacts..."

EXPECTED_FILES=(
  "results/phase3_full_comparison.csv"
  "results/ablation_phase3.csv"
  "results/ablation_phase3_deltas.csv"
  "results/modelC_scores.npz"
  "figures/diagram_model_A.png"
  "figures/diagram_model_C.png"
)

missing=0
for artifact in "${EXPECTED_FILES[@]}"; do
  if [[ -f "$artifact" ]]; then
    echo "OK: ${artifact}"
  else
    echo "MISSING: ${artifact}"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Phase 3 pipeline failed verification: one or more required artifacts are missing."
  exit 1
fi

echo
echo "Phase 3 pipeline completed successfully. All required artifacts were generated."
