"""Data loading, preprocessing, and splitting utilities."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, COLUMN_NAMES, ATTACK_MAPPING, CATEGORICAL_COLS, DROP_COLS, SEED


def load_nslkdd():
    """Load and preprocess the NSL-KDD dataset.

    Returns:
        df_train, df_test: preprocessed DataFrames
        feature_cols: list of feature column names
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
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]]).unique()
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

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


def create_splits(df_train, df_test, feature_cols):
    """Create semi-supervised train/val/test splits.

    Train: 80% of normal data (model fitting)
    Val mixed: 20% normal + attack samples (threshold tuning)
    Test: held-out test set (final evaluation only)

    Returns dict with all arrays and the fitted scaler.
    """
    normal_mask = df_train['binary_label'] == 0
    attack_mask = df_train['binary_label'] == 1

    X_normal = df_train.loc[normal_mask, feature_cols].values
    X_attack_train = df_train.loc[attack_mask, feature_cols].values
    X_test = df_test[feature_cols].values
    y_test = df_test['binary_label'].values
    test_attack_cats = df_test['attack_cat'].values

    X_train_normal, X_val_normal = train_test_split(
        X_normal, test_size=0.2, random_state=SEED)

    np.random.seed(SEED)
    n_val_attack = min(len(X_val_normal), len(X_attack_train))
    val_attack_idx = np.random.choice(len(X_attack_train), size=n_val_attack, replace=False)
    X_val_attack = X_attack_train[val_attack_idx]

    X_val_mixed = np.vstack([X_val_normal, X_val_attack])
    y_val_mixed = np.array([0] * len(X_val_normal) + [1] * len(X_val_attack))
    shuffle_idx = np.random.permutation(len(X_val_mixed))
    X_val_mixed = X_val_mixed[shuffle_idx]
    y_val_mixed = y_val_mixed[shuffle_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_val_normal_scaled = scaler.transform(X_val_normal)
    X_val_mixed_scaled = scaler.transform(X_val_mixed)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train_scaled': X_train_scaled,
        'X_val_normal_scaled': X_val_normal_scaled,
        'X_val_mixed_scaled': X_val_mixed_scaled,
        'X_test_scaled': X_test_scaled,
        'y_val_mixed': y_val_mixed,
        'y_test': y_test,
        'test_attack_cats': test_attack_cats,
        'scaler': scaler,
        'input_dim': len(feature_cols),
    }


def setup_curriculum(X_train_scaled):
    """Setup curriculum learning stages based on L2 distance from centroid.

    Returns:
        curriculum_stages: list of (name, indices) tuples
        epochs_per_stage: list of epoch counts per stage
        distances: distance array for all samples
    """
    centroid = X_train_scaled.mean(axis=0)
    distances = np.linalg.norm(X_train_scaled - centroid, axis=1)
    difficulty_order = np.argsort(distances)

    n = len(X_train_scaled)
    curriculum_stages = [
        ("Easy — closest 33%", difficulty_order[:n // 3]),
        ("Medium — closest 66%", difficulty_order[:2 * n // 3]),
        ("All — 100%", difficulty_order),
    ]
    epochs_per_stage = [15, 15, 20]

    return curriculum_stages, epochs_per_stage, distances
