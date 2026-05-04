"""Streamlit demo for the Phase 3 hybrid IDS.

Loads the trained AE + XGBoost classifier (Model C) and the latent-space
Isolation Forest (Model A), and lets a user paste a single NSL-KDD record (or
pick a sample from the test set) and inspect:

  * Model A — Deep Isolation Forest anomaly score and decision.
  * Model C — predicted attack family with full probability vector and the
    top-driving features for the decision (raw / latent / residual).

Run from the Phase 3 directory:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Allow `from utils...` when invoked via `streamlit run app/streamlit_app.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
if _PHASE3_ROOT not in sys.path:
    sys.path.insert(0, _PHASE3_ROOT)

from utils.config import (
    RESULTS_DIR, MODELS_DIR, DEVICE, ATTACK_CLASSES,
)
from utils.data_loader import load_splits
from utils.hybrid_models import (
    AutoencoderSkip, encode_dataset, recon_error,
    DeepIsolationForest, CascadedAE_XGB,
)


st.set_page_config(page_title='Phase 3 — Hybrid IDS Demo', layout='wide')


@st.cache_resource(show_spinner=True)
def load_artifacts():
    splits = load_splits(os.path.join(RESULTS_DIR, 'processed_data.npz'))
    ae = AutoencoderSkip(dim=splits['input_dim'], latent_dim=32)
    ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'phase3_ae.pth'),
                                  map_location=DEVICE))
    ae.to(DEVICE).eval()

    iforest = joblib.load(os.path.join(MODELS_DIR, 'phase3_iforest_latent.pkl'))
    dif = DeepIsolationForest(ae, DEVICE)
    dif.iforest = iforest
    dif.fitted = True

    xgb = joblib.load(os.path.join(MODELS_DIR, 'phase3_xgb_full.pkl'))
    cax = CascadedAE_XGB(ae, DEVICE,
                         use_raw=True, use_latent=True, use_residuals=True)
    cax.xgb = xgb
    cax.fitted = True

    # Anomaly thresholds saved from notebook 03.
    scores_a = np.load(os.path.join(RESULTS_DIR, 'modelA_scores.npz'))
    return splits, ae, dif, cax, float(scores_a['dif_threshold']), float(scores_a['ae_threshold'])


splits, ae, dif, cax, dif_threshold, ae_threshold = load_artifacts()
feature_cols = splits['feature_cols']
X_test = splits['X_test_scaled']
y_test = splits['y_test']
test_attack_cats = splits['test_attack_cats']

st.title('Hybrid Intrusion Detection — Phase 3 Demo')
st.caption('Models A (Deep Isolation Forest) and C (Cascaded AE + XGBoost) '
           'evaluated live on NSL-KDD records.')

with st.sidebar:
    st.header('Pick an input')
    mode = st.radio('Source',
                    ['Random test sample',
                     'Pick by attack family',
                     'Custom row index'])

    if mode == 'Random test sample':
        if st.button('Resample'):
            st.session_state.idx = int(np.random.randint(0, len(X_test)))
        idx = st.session_state.get('idx', 0)
    elif mode == 'Pick by attack family':
        fam = st.selectbox('Family', ATTACK_CLASSES)
        candidates = np.where(test_attack_cats == fam)[0]
        if len(candidates) == 0:
            st.warning(f'No {fam} samples in test set.')
            idx = 0
        else:
            idx = int(np.random.choice(candidates))
    else:
        idx = st.number_input('Test row index', min_value=0,
                              max_value=len(X_test) - 1, value=0, step=1)

    st.markdown(f'**Row {idx}**  '
                f'— ground truth: `{test_attack_cats[idx]}` '
                f'(binary = {y_test[idx]})')

x = X_test[idx:idx + 1]

# ---------- Model A inference ----------------------------------------------
score_a = float(dif.score(x)[0])
flag_a = score_a > dif_threshold

# ---------- Model C inference ----------------------------------------------
proba = cax.predict_proba_multi(x)[0]
pred_class = int(proba.argmax())
pred_label = ATTACK_CLASSES[pred_class]

# ---------- Per-block contribution (lightweight, not full SHAP) ------------
z, recon = encode_dataset(ae, x, DEVICE)
res = np.abs(x - recon)[0]
ae_mse = float(recon_error(x, recon)[0])

col1, col2 = st.columns(2)

with col1:
    st.subheader('Model A — Deep Isolation Forest')
    st.metric('Anomaly score', f'{score_a:.4f}',
              delta=f'threshold = {dif_threshold:.4f}',
              delta_color='off')
    if flag_a:
        st.error('Decision: ANOMALY')
    else:
        st.success('Decision: NORMAL')
    st.caption(f'AE reconstruction MSE on this row: **{ae_mse:.4f}** '
               f'(AE-only threshold: {ae_threshold:.4f})')

with col2:
    st.subheader('Model C — Cascaded AE + XGBoost')
    proba_df = pd.DataFrame({'class': ATTACK_CLASSES,
                             'probability': proba}).set_index('class')
    st.bar_chart(proba_df)
    if pred_class == 0:
        st.success(f'Decision: NORMAL (P = {proba[0]:.3f})')
    else:
        st.error(f'Decision: {pred_label} (P = {proba[pred_class]:.3f})')

st.divider()

st.subheader('Top per-feature reconstruction residuals (DL → ML signal)')
res_df = (pd.DataFrame({'feature': feature_cols, '|x − x_hat|': res})
          .sort_values('|x − x_hat|', ascending=False)
          .head(15)
          .reset_index(drop=True))
st.dataframe(res_df, use_container_width=True, hide_index=True)
st.caption('These are the dimensions the autoencoder failed to reconstruct — '
           'they are the residual block fed into XGBoost in Model C and explain '
           'much of the hybrid\'s gain over a raw-features-only classifier.')

with st.expander('Raw scaled feature vector'):
    st.dataframe(pd.DataFrame({'feature': feature_cols, 'value': x[0]}),
                 use_container_width=True, hide_index=True)
