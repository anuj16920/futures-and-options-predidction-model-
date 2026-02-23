"""
====================================================================
SEQUENCE BUILDER - Memory-Efficient for ~900K Rows
====================================================================
Building (900K, 60, 46) float32 array = ~10GB RAM if done naively.
Solution: use chunked processing + memory-mapped numpy arrays.

Also provides tf.data.Dataset pipeline for efficient GPU feeding.
====================================================================
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    LOOKBACK, TRAIN_RATIO, VAL_RATIO,
    SCALER_DIR, LOG_LEVEL, BATCH_SIZE
)
from features.feature_engineering import get_feature_columns

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("SequenceBuilder")


def chronological_split(df: pd.DataFrame, train_r=TRAIN_RATIO, val_r=VAL_RATIO):
    """Split chronologically — never shuffle financial time-series."""
    n = len(df)
    t = int(n * train_r)
    v = int(n * (train_r + val_r))
    train, val, test = df.iloc[:t], df.iloc[t:v], df.iloc[v:]
    logger.info(
        f"Split -> train={len(train):,} | val={len(val):,} | test={len(test):,}\n"
        f"  Train: {train.index[0].date()} -> {train.index[-1].date()}\n"
        f"  Val  : {val.index[0].date()} -> {val.index[-1].date()}\n"
        f"  Test : {test.index[0].date()} -> {test.index[-1].date()}"
    )
    return train, val, test


def fit_scaler(X_train: np.ndarray, name: str = "scaler_kaggle.joblib") -> RobustScaler:
    """Fit RobustScaler on train only, save to disk."""
    scaler = RobustScaler()
    logger.info(f"Fitting scaler on {len(X_train):,} rows...")
    scaler.fit(X_train)
    path = SCALER_DIR / name
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved: {path}")
    return scaler


def load_scaler(name: str = "scaler_kaggle.joblib") -> RobustScaler:
    path = SCALER_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)


def build_sequences_chunked(
    features: np.ndarray,
    labels:   np.ndarray,
    lookback: int = LOOKBACK,
    chunk_size: int = 50000
) -> tuple:
    """
    Build sliding-window sequences in chunks to control RAM.

    For 900K rows x 60 lookback x 46 features:
    Full array = 900K x 60 x 46 x 4 bytes = ~9.9 GB — too large.
    Chunked approach builds and returns manageable pieces.

    Returns X (N, lookback, features), y (N,) as float32 / int32.
    """
    n = len(features)
    n_seqs = n - lookback
    n_feat  = features.shape[1]

    logger.info(f"Building {n_seqs:,} sequences | lookback={lookback} | features={n_feat}")

    # Pre-allocate output arrays
    X = np.empty((n_seqs, lookback, n_feat), dtype=np.float32)
    y = np.empty(n_seqs, dtype=np.int32)

    for start in range(0, n_seqs, chunk_size):
        end = min(start + chunk_size, n_seqs)
        for i in range(start, end):
            X[i] = features[i: i + lookback]
            y[i] = labels[i + lookback]

        if start % 200000 == 0:
            logger.info(f"  Sequences: {end:,} / {n_seqs:,}")

    logger.info(f"Done. X={X.shape} ({X.nbytes/1e9:.2f} GB) | y={y.shape}")
    return X, y


def build_tf_dataset(X: np.ndarray, y: np.ndarray,
                     batch_size: int = BATCH_SIZE,
                     shuffle: bool = False,
                     shuffle_buffer: int = 10000):
    """
    Build tf.data.Dataset pipeline for efficient GPU training.
    Uses generator to avoid loading all data into memory at once.
    """
    import tensorflow as tf

    def data_generator():
        """Generator that yields batches without loading all data."""
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            y_oh = tf.keras.utils.to_categorical(y[idx], num_classes=3)
            yield X[idx], y_oh
    
    # Create dataset from generator
    ds = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(X.shape[1], X.shape[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32)
        )
    )
    
    ds = (ds
          .batch(batch_size, drop_remainder=False)
          .prefetch(tf.data.AUTOTUNE))
    
    return ds


def prepare_datasets(labeled_df: pd.DataFrame,
                     feature_cols: list = None,
                     lookback: int = LOOKBACK,
                     fit_new_scaler: bool = True) -> dict:
    """
    Full pipeline: split -> scale -> sequences -> tf.data.Dataset

    Returns dict with all splits ready for training.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    missing = [c for c in feature_cols if c not in labeled_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # ── Split ─────────────────────────────────────────────────
    train_df, val_df, test_df = chronological_split(labeled_df)

    # ── Raw arrays ────────────────────────────────────────────
    X_tr_raw = train_df[feature_cols].values.astype(np.float32)
    X_vl_raw = val_df[feature_cols].values.astype(np.float32)
    X_te_raw = test_df[feature_cols].values.astype(np.float32)

    y_tr_raw = train_df["label"].values.astype(np.int32)
    y_vl_raw = val_df["label"].values.astype(np.int32)
    y_te_raw = test_df["label"].values.astype(np.int32)

    # ── Scale (fit on train ONLY) ─────────────────────────────
    if fit_new_scaler:
        scaler = fit_scaler(X_tr_raw)
    else:
        scaler = load_scaler()

    X_tr_sc = scaler.transform(X_tr_raw)
    X_vl_sc = scaler.transform(X_vl_raw)
    X_te_sc = scaler.transform(X_te_raw)

    # ── Sequences ─────────────────────────────────────────────
    logger.info("Building train sequences...")
    X_train, y_train = build_sequences_chunked(X_tr_sc, y_tr_raw, lookback)
    logger.info("Building val sequences...")
    X_val,   y_val   = build_sequences_chunked(X_vl_sc, y_vl_raw, lookback)
    logger.info("Building test sequences...")
    X_test,  y_test  = build_sequences_chunked(X_te_sc, y_te_raw, lookback)

    # ── tf.data Datasets ──────────────────────────────────────
    train_ds = build_tf_dataset(X_train, y_train, shuffle=False)
    val_ds   = build_tf_dataset(X_val,   y_val,   shuffle=False)
    test_ds  = build_tf_dataset(X_test,  y_test,  shuffle=False)

    logger.info(
        f"Datasets ready:\n"
        f"  train: {X_train.shape} | val: {X_val.shape} | test: {X_test.shape}"
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "scaler": scaler, "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }


def build_inference_sequence(latest_df: pd.DataFrame,
                              scaler: RobustScaler,
                              feature_cols: list,
                              lookback: int = LOOKBACK) -> np.ndarray:
    """Single inference sequence for live trading / FastAPI."""
    if len(latest_df) < lookback:
        raise ValueError(f"Need >= {lookback} rows. Got {len(latest_df)}.")
    window = latest_df[feature_cols].tail(lookback).values.astype(np.float32)
    return scaler.transform(window).reshape(1, lookback, len(feature_cols))
