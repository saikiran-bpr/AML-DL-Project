"""Data loading, leakage-safe splitting, and multi-class label preparation."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .config import (
    DATA_DIR, COLUMN_NAMES, ATTACK_MAPPING, CATEGORICAL_COLS, DROP_COLS,
    SEED, ATTACK_TO_IDX,
)


def load_nslkdd():
    """Load NSL-KDD train/test files and produce engineered tabular features.

    Categorical encoders are fit on training data only; unseen test categories
    are mapped to -1 to avoid distribution leakage.
    """
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'KDDTrain+.txt'),
                           header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'KDDTest+.txt'),
                          header=None, names=COLUMN_NAMES)

    df_train['attack_cat'] = df_train['label'].map(ATTACK_MAPPING).fillna('Unknown')
    df_test['attack_cat'] = df_test['label'].map(ATTACK_MAPPING).fillna('Unknown')
    df_train['binary_label'] = (df_train['label'] != 'normal').astype(int)
    df_test['binary_label'] = (df_test['label'] != 'normal').astype(int)

    for col in CATEGORICAL_COLS:
        train_categories = sorted(df_train[col].astype(str).unique().tolist())
        category_to_index = {cat: idx for idx, cat in enumerate(train_categories)}
        df_train[col] = df_train[col].astype(str).map(category_to_index).astype(int)
        df_test[col] = (
            df_test[col].astype(str).map(category_to_index).fillna(-1).astype(int)
        )

    feature_cols = [c for c in df_train.columns if c not in DROP_COLS]
    feature_cols = [c for c in feature_cols if df_train[c].std() > 0]

    df_train['bytes_ratio'] = df_train['src_bytes'] / (df_train['dst_bytes'] + 1)
    df_test['bytes_ratio'] = df_test['src_bytes'] / (df_test['dst_bytes'] + 1)
    df_train['error_rate_diff'] = df_train['serror_rate'] - df_train['srv_serror_rate']
    df_test['error_rate_diff'] = df_test['serror_rate'] - df_test['srv_serror_rate']
    df_train['srv_diversity'] = df_train['diff_srv_rate'] / (df_train['same_srv_rate'] + 1e-6)
    df_test['srv_diversity'] = df_test['diff_srv_rate'] / (df_test['same_srv_rate'] + 1e-6)

    feature_cols.extend(['bytes_ratio', 'error_rate_diff', 'srv_diversity'])
    return df_train, df_test, feature_cols


def attack_cat_to_idx(cats):
    """Map attack-category strings to integer class indices, with Unknown→Normal-fallback."""
    return np.array([ATTACK_TO_IDX.get(c, ATTACK_TO_IDX['Normal']) for c in cats])


def create_hybrid_splits(df_train, df_test, feature_cols):
    """Build the leakage-safe split set used by both Model A and Model C.

    Splits produced:
      - X_train_normal_scaled : 80% of train normals — AE training only.
      - X_val_normal_scaled   : remaining train normals — threshold tuning.
      - X_val_mixed_scaled / y_val_mixed : val_normal + sampled attacks for
        binary threshold tuning (matches Phase 2's protocol).
      - X_clf_train_scaled / y_clf_train_multi : the held-out val_normal plus
        ALL train attacks, with attack-family multi-class labels — used by the
        XGBoost classifier in Model C. Critically, this never overlaps with
        AE's training rows.
      - X_test_scaled / y_test / y_test_multi / test_attack_cats — final eval.

    The scaler is fit on AE-train normals only and re-used everywhere.
    """
    normal_mask = df_train['binary_label'] == 0
    attack_mask = df_train['binary_label'] == 1

    X_normal = df_train.loc[normal_mask, feature_cols].values
    X_attack_train = df_train.loc[attack_mask, feature_cols].values
    attack_cats_train = df_train.loc[attack_mask, 'attack_cat'].values

    X_test = df_test[feature_cols].values
    y_test = df_test['binary_label'].values
    test_attack_cats = df_test['attack_cat'].values
    y_test_multi = attack_cat_to_idx(test_attack_cats)

    X_train_normal, X_val_normal = train_test_split(
        X_normal, test_size=0.2, random_state=SEED)

    rng = np.random.default_rng(SEED)
    n_val_attack = min(len(X_val_normal), len(X_attack_train))
    val_attack_idx = rng.choice(len(X_attack_train), size=n_val_attack, replace=False)
    X_val_attack = X_attack_train[val_attack_idx]

    X_val_mixed = np.vstack([X_val_normal, X_val_attack])
    y_val_mixed = np.array([0] * len(X_val_normal) + [1] * len(X_val_attack))
    shuffle_idx = rng.permutation(len(X_val_mixed))
    X_val_mixed = X_val_mixed[shuffle_idx]
    y_val_mixed = y_val_mixed[shuffle_idx]

    # XGBoost training pool: held-out val_normal + every train attack.
    # AE never sees these normals during training, so there's no representational
    # leakage when we feed AE-derived features (z, residuals) into XGBoost.
    X_clf_train = np.vstack([X_val_normal, X_attack_train])
    y_clf_train_multi = np.concatenate([
        np.full(len(X_val_normal), ATTACK_TO_IDX['Normal']),
        attack_cat_to_idx(attack_cats_train),
    ])
    clf_shuffle = rng.permutation(len(X_clf_train))
    X_clf_train = X_clf_train[clf_shuffle]
    y_clf_train_multi = y_clf_train_multi[clf_shuffle]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_val_normal_scaled = scaler.transform(X_val_normal)
    X_val_mixed_scaled = scaler.transform(X_val_mixed)
    X_clf_train_scaled = scaler.transform(X_clf_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train_normal_scaled': X_train_scaled,
        'X_val_normal_scaled': X_val_normal_scaled,
        'X_val_mixed_scaled': X_val_mixed_scaled,
        'y_val_mixed': y_val_mixed,
        'X_clf_train_scaled': X_clf_train_scaled,
        'y_clf_train_multi': y_clf_train_multi,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_test_multi': y_test_multi,
        'test_attack_cats': test_attack_cats,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'input_dim': len(feature_cols),
    }


def save_splits(splits, path):
    """Persist splits as a single .npz so notebooks 03–06 share identical data."""
    payload = {k: v for k, v in splits.items()
               if isinstance(v, np.ndarray)}
    payload['feature_cols'] = np.array(splits['feature_cols'], dtype=object)
    np.savez_compressed(path, **payload)


def load_splits(path):
    """Reload the splits saved by save_splits()."""
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out['feature_cols'] = list(out['feature_cols'])
    out['input_dim'] = len(out['feature_cols'])
    return out
