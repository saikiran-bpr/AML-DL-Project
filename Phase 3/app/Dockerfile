# Reproducible container for the Phase 3 hybrid IDS demo.
# Build from the repository root:
#   docker build -t phase3-ids -f "Phase 3/app/Dockerfile" .
# Run:
#   docker run --rm -p 8501:8501 phase3-ids
#
# The build context must be the *project root* (so the container can also see
# the Phase 1 NSL-KDD data files at /data/Phase\ 1/data/).

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libs needed by xgboost wheels and matplotlib font rendering.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for caching.
COPY "Phase 3/requirements.txt" /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the project (Phase 1 data + Phase 3 source).
COPY "Phase 1/data" "/data/Phase 1/data"
COPY "Phase 3" "/app/Phase 3"

WORKDIR "/app/Phase 3"

EXPOSE 8501

# The app expects splits + trained models in results/ and models/. If they are
# missing the entrypoint regenerates them by executing notebooks 02 → 04.
CMD ["bash", "-lc", "\
    if [ ! -f results/processed_data.npz ] || [ ! -f models/phase3_ae.pth ] || [ ! -f models/phase3_xgb_full.pkl ]; then \
      jupyter nbconvert --to notebook --execute 02_data_preparation.ipynb --output 02_data_preparation.ipynb && \
      jupyter nbconvert --to notebook --execute 03_deep_isolation_forest.ipynb --output 03_deep_isolation_forest.ipynb && \
      jupyter nbconvert --to notebook --execute 04_cascaded_ae_xgboost.ipynb --output 04_cascaded_ae_xgboost.ipynb ; \
    fi && \
    streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 \
"]
