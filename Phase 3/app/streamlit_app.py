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
import matplotlib.pyplot as plt

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


st.set_page_config(
    page_title='Hybrid IDS Dashboard',
    page_icon='shield',
    layout='wide',
    initial_sidebar_state='expanded',
)


CSS = """
<style>
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes glowPulse {
  0% { box-shadow: 0 0 0 rgba(99, 102, 241, 0.0); }
  50% { box-shadow: 0 0 28px rgba(99, 102, 241, 0.22); }
  100% { box-shadow: 0 0 0 rgba(99, 102, 241, 0.0); }
}

:root {
  --bg: #080b12;
  --panel: rgba(17, 24, 39, 0.78);
  --panel-strong: rgba(15, 23, 42, 0.94);
  --border: rgba(148, 163, 184, 0.18);
  --text: #f8fafc;
  --muted: #94a3b8;
  --blue: #38bdf8;
  --purple: #a78bfa;
  --green: #22c55e;
  --red: #ef4444;
  --yellow: #f59e0b;
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 32rem),
    radial-gradient(circle at top right, rgba(167, 139, 250, 0.16), transparent 28rem),
    linear-gradient(135deg, #070a12 0%, #0f172a 48%, #111827 100%);
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.96));
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--text);
}

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 1.2rem;
  max-width: 1420px;
}

.hero {
  animation: fadeInUp 520ms ease-out;
  padding: 1.35rem 1.45rem;
  border: 1px solid rgba(148, 163, 184, 0.20);
  border-radius: 22px;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.72)),
    linear-gradient(90deg, rgba(56, 189, 248, 0.20), rgba(167, 139, 250, 0.18));
  box-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
  margin-bottom: 1rem;
}

.hero-title {
  font-size: clamp(2.1rem, 4vw, 4.4rem);
  line-height: 1.02;
  font-weight: 850;
  letter-spacing: 0;
  margin: 0;
}

.hero-subtitle {
  margin: 0.65rem 0 1rem;
  color: var(--muted);
  font-size: 1.03rem;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(15, 23, 42, 0.66);
  color: #dbeafe;
  padding: 0.42rem 0.68rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
}

.glass-card {
  animation: fadeInUp 620ms ease-out;
  border: 1px solid var(--border);
  border-radius: 20px;
  background: var(--panel);
  box-shadow: 0 12px 42px rgba(0, 0, 0, 0.26);
  padding: 1.05rem 1.1rem;
  backdrop-filter: blur(14px);
  transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
}

.glass-card:hover {
  transform: translateY(-2px);
  border-color: rgba(125, 211, 252, 0.38);
  box-shadow: 0 18px 52px rgba(0, 0, 0, 0.34);
}

.kpi-card {
  min-height: 7.25rem;
}

.kpi-label {
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.45rem;
}

.kpi-value {
  color: var(--text);
  font-size: 1.55rem;
  font-weight: 850;
  line-height: 1.12;
}

.kpi-note {
  color: var(--muted);
  margin-top: 0.55rem;
  font-size: 0.82rem;
}

.status-safe { color: var(--green); }
.status-attack { color: var(--red); }
.status-warn { color: var(--yellow); }
.status-model { color: var(--blue); }

.decision-card {
  animation: glowPulse 3.6s ease-in-out infinite;
}

.section-title {
  margin: 1rem 0 0.45rem;
  font-size: 1.25rem;
  font-weight: 850;
}

.section-subtitle {
  margin: -0.2rem 0 0.8rem;
  color: var(--muted);
  font-size: 0.92rem;
}

.decision-pill {
  display: inline-flex;
  padding: 0.48rem 0.72rem;
  border-radius: 999px;
  font-weight: 850;
  border: 1px solid rgba(255,255,255,0.12);
}

.pill-safe {
  background: rgba(34, 197, 94, 0.14);
  color: #86efac;
  border-color: rgba(34, 197, 94, 0.36);
}

.pill-attack {
  background: rgba(239, 68, 68, 0.14);
  color: #fca5a5;
  border-color: rgba(239, 68, 68, 0.36);
}

.flow-row {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr auto 1fr;
  gap: 0.45rem;
  align-items: center;
}

.flow-box {
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.72);
  border-radius: 16px;
  padding: 0.78rem;
  text-align: center;
  min-height: 4.5rem;
}

.flow-box strong {
  display: block;
  color: #e0f2fe;
}

.flow-box span {
  color: var(--muted);
  font-size: 0.78rem;
}

.flow-arrow {
  color: #7dd3fc;
  font-weight: 900;
  font-size: 1.3rem;
}

.sidebar-card {
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.64);
  border-radius: 16px;
  padding: 0.85rem 0.9rem;
  margin: 0.65rem 0;
}

.sidebar-meta {
  color: #dbeafe;
  font-size: 0.88rem;
  line-height: 1.7;
}

.footer {
  margin-top: 1.25rem;
  padding: 0.9rem;
  color: var(--muted);
  text-align: center;
  border-top: 1px solid var(--border);
  font-size: 0.88rem;
}

.error-card {
  border: 1px solid rgba(239, 68, 68, 0.40);
  background: rgba(127, 29, 29, 0.22);
  color: #fecaca;
  border-radius: 18px;
  padding: 1rem 1.1rem;
}

@media (max-width: 900px) {
  .flow-row {
    grid-template-columns: 1fr;
  }
  .flow-arrow {
    transform: rotate(90deg);
    text-align: center;
  }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def html_escape(value):
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def format_class(label):
    return html_escape(label)


def status_class(is_attack=False, warn=False, model=False):
    if is_attack:
        return 'status-attack'
    if warn:
        return 'status-warn'
    if model:
        return 'status-model'
    return 'status-safe'


def decision_pill(label, is_attack):
    cls = 'pill-attack' if is_attack else 'pill-safe'
    return f'<span class="decision-pill {cls}">{html_escape(label)}</span>'


def kpi_card(label, value, note='', color_class='status-model'):
    st.markdown(
        f"""
        <div class="glass-card kpi-card">
          <div class="kpi-label">{html_escape(label)}</div>
          <div class="kpi-value {color_class}">{html_escape(value)}</div>
          <div class="kpi-note">{html_escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


@st.cache_resource(show_spinner=True)
def load_artifacts():
    processed_path = os.path.join(RESULTS_DIR, 'processed_data.npz')
    ae_path = os.path.join(MODELS_DIR, 'phase3_ae.pth')
    iforest_path = os.path.join(MODELS_DIR, 'phase3_iforest_latent.pkl')
    xgb_path = os.path.join(MODELS_DIR, 'phase3_xgb_full.pkl')
    scores_path = os.path.join(RESULTS_DIR, 'modelA_scores.npz')

    for path in (processed_path, ae_path, iforest_path, xgb_path, scores_path):
        require_file(path)

    splits = load_splits(processed_path)
    ae = AutoencoderSkip(dim=splits['input_dim'], latent_dim=32)
    ae.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    ae.to(DEVICE).eval()

    iforest = joblib.load(iforest_path)
    dif = DeepIsolationForest(ae, DEVICE)
    dif.iforest = iforest
    dif.fitted = True

    xgb = joblib.load(xgb_path)
    cax = CascadedAE_XGB(ae, DEVICE,
                         use_raw=True, use_latent=True, use_residuals=True)
    cax.xgb = xgb
    cax.fitted = True

    # Anomaly thresholds saved from notebook 03.
    scores_a = np.load(scores_path)
    return splits, ae, dif, cax, float(scores_a['dif_threshold']), float(scores_a['ae_threshold'])


try:
    splits, ae, dif, cax, dif_threshold, ae_threshold = load_artifacts()
except FileNotFoundError as exc:
    st.markdown(
        f"""
        <div class="error-card">
          <h3>Required artifact missing</h3>
          <p>The dashboard could not find this file:</p>
          <code>{html_escape(exc.filename or exc.args[0])}</code>
          <p>Run <code>./run_all.sh</code> from the <code>Phase 3</code> directory to regenerate artifacts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
except Exception as exc:
    st.markdown(
        f"""
        <div class="error-card">
          <h3>Dashboard initialization failed</h3>
          <p>{html_escape(type(exc).__name__)}: {html_escape(exc)}</p>
          <p>Check that the generated Phase-3 artifacts match the current code.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

feature_cols = splits['feature_cols']
X_test = splits['X_test_scaled']
y_test = splits['y_test']
if 'test_attack_cats' in splits:
    test_attack_cats = splits['test_attack_cats']
else:
    y_test_multi = splits['y_test_multi']
    test_attack_cats = np.array([ATTACK_CLASSES[int(i)] for i in y_test_multi])

st.markdown(
    """
    <section class="hero">
      <h1 class="hero-title">Hybrid Intrusion Detection Dashboard</h1>
      <p class="hero-subtitle">Phase 3 ML + DL Intrusion Detection Demo</p>
      <div class="badge-row">
        <span class="badge">Model A: Deep Isolation Forest</span>
        <span class="badge">Model C: Cascaded AE + XGBoost</span>
        <span class="badge">Dataset: NSL-KDD</span>
        <span class="badge">Live Inference</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('## Input Control Panel')
    st.caption('Choose a test-set record and inspect both hybrid IDS models live.')
    st.markdown(
        """
        <div class="sidebar-card">
          <strong>Source selection</strong><br>
          <span style="color:#94a3b8;font-size:0.86rem;">
          Random samples are useful for demos; attack-family mode lets evaluators
          probe specific NSL-KDD categories.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        'Input source',
        ['Random test sample', 'Pick by attack family', 'Custom row index'],
        label_visibility='collapsed',
    )

    if mode == 'Random test sample':
        st.caption('Draw a record from the held-out NSL-KDD test split.')
        if st.button('Resample record', use_container_width=True):
            st.session_state.idx = int(np.random.randint(0, len(X_test)))
        idx = st.session_state.get('idx', 0)
    elif mode == 'Pick by attack family':
        st.caption('Select an attack family, then sample one test record from it.')
        fam = st.selectbox('Attack family', ATTACK_CLASSES)
        candidates = np.where(test_attack_cats == fam)[0]
        if len(candidates) == 0:
            st.warning(f'No {fam} samples in test set.')
            idx = 0
        else:
            if st.button('Sample from family', use_container_width=True):
                st.session_state.family_idx = int(np.random.choice(candidates))
            idx = st.session_state.get('family_idx', int(candidates[0]))
    else:
        st.caption('Enter a specific test-set row index for repeatable inspection.')
        idx = st.number_input('Test row index', min_value=0,
                              max_value=len(X_test) - 1, value=0, step=1)

    truth_label = str(test_attack_cats[idx])
    truth_binary = int(y_test[idx])
    truth_kind = 'Attack' if truth_binary else 'Normal'
    st.markdown(
        f"""
        <div class="sidebar-card sidebar-meta">
          <strong>Selected row</strong><br>
          Row index: <code>{idx}</code><br>
          Ground truth: <code>{format_class(truth_label)}</code><br>
          Binary label: <code>{truth_binary}</code> ({truth_kind})<br>
          Test samples available: <code>{len(X_test):,}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

x = X_test[idx:idx + 1]

# ---------- Model A inference ----------------------------------------------
score_a = float(dif.score(x)[0])
flag_a = score_a > dif_threshold

# ---------- Model C inference ----------------------------------------------
proba = cax.predict_proba_multi(x)[0]
pred_class = int(proba.argmax())
pred_label = ATTACK_CLASSES[pred_class]
pred_confidence = float(proba[pred_class])

# ---------- Per-block contribution (lightweight, not full SHAP) ------------
z, recon = encode_dataset(ae, x, DEVICE)
res = np.abs(x - recon)[0]
ae_mse = float(recon_error(x, recon)[0])

model_a_decision = 'ATTACK' if flag_a else 'NORMAL'
model_c_decision = pred_label.upper()
model_c_attack = pred_class != 0
confidence_color = status_class(warn=pred_confidence < 0.70,
                                model=pred_confidence >= 0.70)

kpi_cols = st.columns(5)
with kpi_cols[0]:
    kpi_card('Ground Truth', truth_label, f'Binary: {truth_kind}',
             status_class(is_attack=truth_binary == 1))
with kpi_cols[1]:
    kpi_card('Model A Decision', model_a_decision, f'Score {score_a:.4f}',
             status_class(is_attack=flag_a))
with kpi_cols[2]:
    kpi_card('Model C Decision', model_c_decision, f'Class: {pred_label}',
             status_class(is_attack=model_c_attack))
with kpi_cols[3]:
    kpi_card('AE Reconstruction MSE', f'{ae_mse:.5f}',
             f'AE threshold {ae_threshold:.5f}',
             status_class(warn=ae_mse > ae_threshold, model=ae_mse <= ae_threshold))
with kpi_cols[4]:
    kpi_card('Model C Confidence', f'{pred_confidence:.3f}',
             'Top class probability', confidence_color)

st.markdown('<div class="section-title">Hybrid Architecture Flow</div>',
            unsafe_allow_html=True)
st.markdown(
    """
    <div class="glass-card">
      <div class="flow-row">
        <div class="flow-box"><strong>Raw Features</strong><span>NSL-KDD scaled row</span></div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-box"><strong>Autoencoder</strong><span>normal manifold</span></div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-box"><strong>Latent + Residuals</strong><span>z and |x - x_hat|</span></div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-box"><strong>XGBoost</strong><span>class probabilities</span></div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-box"><strong>Prediction</strong><span>Normal / attack family</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Model Comparison</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Side-by-side scoring from the unsupervised hybrid detector and the supervised cascaded classifier.</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown('<div class="glass-card decision-card">', unsafe_allow_html=True)
    st.markdown('### Model A: Deep Isolation Forest')
    st.metric('Anomaly score', f'{score_a:.4f}',
              delta=f'threshold = {dif_threshold:.4f}',
              delta_color='off')
    st.markdown(decision_pill(f'Decision: {model_a_decision}', flag_a),
                unsafe_allow_html=True)
    st.caption(
        f'AE reconstruction MSE on this row: {ae_mse:.5f} '
        f'(AE-only threshold: {ae_threshold:.5f})'
    )
    st.progress(min(max(score_a / max(dif_threshold * 1.6, 1e-6), 0.0), 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card decision-card">', unsafe_allow_html=True)
    st.markdown('### Model C: Cascaded AE + XGBoost')
    st.markdown(decision_pill(f'Decision: {pred_label} (P = {pred_confidence:.3f})',
                              model_c_attack),
                unsafe_allow_html=True)
    proba_df = (pd.DataFrame({'class': ATTACK_CLASSES, 'probability': proba})
                .sort_values('probability', ascending=True))

    fig, ax = plt.subplots(figsize=(7.5, 3.25))
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    colors = ['#22c55e' if c == 'Normal' else '#a78bfa'
              for c in proba_df['class']]
    bars = ax.barh(proba_df['class'], proba_df['probability'], color=colors, alpha=0.92)
    for bar, val in zip(bars, proba_df['probability']):
        ax.text(min(val + 0.018, 0.98), bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color='#e5e7eb', fontsize=10,
                fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Probability', color='#cbd5e1')
    ax.tick_params(colors='#cbd5e1')
    ax.grid(axis='x', color='#334155', alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Top Reconstruction Residuals</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Features with highest DL reconstruction error passed into the ML layer.</div>',
    unsafe_allow_html=True,
)

res_df = (pd.DataFrame({'feature': feature_cols, 'residual': res})
          .sort_values('residual', ascending=False)
          .head(10)
          .reset_index(drop=True))

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
plot_df = res_df.sort_values('residual', ascending=True)
fig, ax = plt.subplots(figsize=(9, 3.5))
fig.patch.set_alpha(0)
ax.set_facecolor((0, 0, 0, 0))
bars = ax.barh(plot_df['feature'], plot_df['residual'], color='#38bdf8', alpha=0.88)
for bar, val in zip(bars, plot_df['residual']):
    ax.text(val + max(plot_df['residual'].max() * 0.012, 0.0001),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', color='#e5e7eb', fontsize=9)
ax.set_xlabel('|x - x_hat|', color='#cbd5e1')
ax.tick_params(colors='#cbd5e1', labelsize=9)
ax.grid(axis='x', color='#334155', alpha=0.45)
for spine in ax.spines.values():
    spine.set_visible(False)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.dataframe(
    res_df.rename(columns={'residual': '|x - x_hat|'}),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    'High residuals identify the dimensions the autoencoder failed to reconstruct; '
    'these residual values are part of the feature block consumed by XGBoost.'
)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander('Raw scaled feature vector'):
    st.dataframe(pd.DataFrame({'feature': feature_cols, 'value': x[0]}),
                 use_container_width=True, hide_index=True)

st.markdown(
    '<div class="footer">Phase 3 Hybrid IDS Demo · Rishihood University · '
    'Prerak Arya (230039) · Sai Kiran Bompelliwar (230046)</div>',
    unsafe_allow_html=True,
)
