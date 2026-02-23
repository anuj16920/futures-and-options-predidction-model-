"""
====================================================================
TRAINING PIPELINE - RTX 3050 GPU Optimized
====================================================================
GPU setup is called FIRST before any TF imports.
Uses tf.data pipeline for maximum GPU utilization.
Mixed precision FP16 enabled for ~2x speedup on Ampere.
====================================================================
"""

import json
import logging
import sys
import time
from pathlib import Path

# ── GPU MUST BE CONFIGURED BEFORE TF IMPORTS ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.gpu_config import setup_gpu, get_optimal_batch_size
gpu_available = setup_gpu(
    allow_growth=True,
    mixed_precision=True,   # FP16 on RTX 3050 = ~2x faster
    verbose=True
)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config.config import (
    EPOCHS, BATCH_SIZE, LOOKBACK,
    MODEL_DIR, TENSORBOARD_DIR, LOG_DIR,
    WFV_N_SPLITS, WFV_TRAIN_WINDOW, WFV_STEP_SIZE,
    WARMUP_EPOCHS, LOG_LEVEL
)
from data.loader import load_and_save
from features.feature_engineering import build_features, get_feature_columns
from features.label_generator import create_labels, get_class_weights, validate_labels
from features.sequence_builder import (
    prepare_datasets, build_sequences_chunked,
    fit_scaler, build_tf_dataset
)
from models.architecture import (
    build_model, get_callbacks, print_model_summary,
    FocalLoss, WarmupCosineSchedule, PositionalEncoding, TransformerEncoderBlock
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "training.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("Trainer")


def set_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(force_reload=False, seed=42):
    """Full end-to-end GPU-accelerated training pipeline."""
    set_seeds(seed)
    t0 = time.time()

    # Use GPU-optimal batch size for RTX 3050
    batch_size = get_optimal_batch_size("transformer") if gpu_available else BATCH_SIZE
    logger.info(f"Batch size: {batch_size} (GPU={'YES RTX 3050' if gpu_available else 'NO - CPU'})")

    # ── Load & process data ────────────────────────────────────
    logger.info("STEP 1/6 - Loading Kaggle data")
    raw = load_and_save(force=force_reload)

    logger.info("STEP 2/6 - Engineering multi-timeframe features")
    featured = build_features(raw)
    feat_cols  = get_feature_columns()
    n_features = len(feat_cols)
    logger.info(f"Features: {n_features}")

    logger.info("STEP 3/6 - Generating labels")
    labeled = create_labels(featured)
    validate_labels(labeled)
    class_weights = get_class_weights(labeled["label"])

    logger.info("STEP 4/6 - Building sequences (multi-minute, memory-efficient)")
    data = prepare_datasets(labeled, feat_cols, LOOKBACK, fit_new_scaler=True)

    steps_per_epoch = len(data["X_train"]) // batch_size

    # ── Build model ───────────────────────────────────────────
    logger.info("STEP 5/6 - Building Transformer-CNN-BiLSTM model")

    # Use MirroredStrategy if multiple GPUs available (future-proof)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Multi-GPU training: {len(gpus)} GPUs")
    else:
        strategy = tf.distribute.get_strategy()   # default (single GPU)

    with strategy.scope():
        model = build_model(
            lookback=LOOKBACK,
            n_features=n_features,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=WARMUP_EPOCHS,
            total_epochs=EPOCHS,
        )

    print_model_summary(model)

    # ── Train ─────────────────────────────────────────────────
    logger.info("STEP 6/6 - Training on RTX 3050")

    # Rebuild tf.data with correct batch size
    train_ds = data["train_ds"]
    val_ds   = data["val_ds"]

    ckpt_path = MODEL_DIR / "best_kaggle_model.keras"
    callbacks = get_callbacks(ckpt_path, TENSORBOARD_DIR)

    # GPU utilization monitor callback
    callbacks.append(GPUMonitorCallback(log_every_n_epochs=5))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save ──────────────────────────────────────────────────
    keras_path = MODEL_DIR / "nifty_kaggle_final.keras"
    model.save(str(keras_path))

    saved_path = MODEL_DIR / "nifty_kaggle_savedmodel"
    model.export(str(saved_path))

    # ── Evaluate ──────────────────────────────────────────────
    test_results = model.evaluate(data["test_ds"], verbose=1)
    test_loss, test_acc = test_results[0], test_results[1]

    elapsed = time.time() - t0
    logger.info(
        f"\nTRAINING COMPLETE\n"
        f"  Time      : {elapsed/60:.1f} min\n"
        f"  Test loss : {test_loss:.4f}\n"
        f"  Test acc  : {test_acc:.4f}\n"
        f"  GPU used  : {'RTX 3050' if gpu_available else 'CPU (slow!)'}"
    )

    return {
        "model": model, "history": history.history,
        "data": data, "feat_cols": feat_cols,
        "class_weights": class_weights,
    }


class GPUMonitorCallback(keras.callbacks.Callback):
    """
    Logs GPU VRAM usage every N epochs.
    Helps detect OOM risk on RTX 3050 4GB.
    """
    def __init__(self, log_every_n_epochs=5):
        super().__init__()
        self.n = log_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n != 0:
            return
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                used, total, util = parts[0], parts[1], parts[2]
                print(f"\n  [GPU] VRAM: {used}MB / {total}MB | Util: {util}%")
        except Exception:
            pass


def walk_forward_validation(labeled_df, feature_cols,
                             n_splits=WFV_N_SPLITS,
                             train_window=WFV_TRAIN_WINDOW,
                             step_size=WFV_STEP_SIZE,
                             lookback=LOOKBACK, seed=42):
    """Walk-Forward Validation across N folds on GPU."""
    set_seeds(seed)
    n = len(labeled_df)
    results = []

    batch_size = get_optimal_batch_size("transformer") if gpu_available else BATCH_SIZE

    logger.info(f"WFV | {n_splits} folds | GPU={'RTX 3050' if gpu_available else 'CPU'}")

    for fold in range(n_splits):
        train_end = int(n * (train_window + fold * step_size))
        val_end   = min(int(n * (train_window + (fold+1) * step_size)), n)

        if train_end >= n or val_end <= train_end:
            break

        tr_df = labeled_df.iloc[:train_end]
        vl_df = labeled_df.iloc[train_end:val_end]

        logger.info(
            f"Fold {fold+1}/{n_splits} | "
            f"train={len(tr_df):,} | val={len(vl_df):,}"
        )

        if len(tr_df) < lookback + 200 or len(vl_df) < lookback + 50:
            logger.warning(f"Fold {fold+1}: skipping (insufficient data)")
            continue

        X_tr = tr_df[feature_cols].values.astype(np.float32)
        X_vl = vl_df[feature_cols].values.astype(np.float32)
        scaler = fit_scaler(X_tr, f"scaler_wfv_fold{fold+1}.joblib")
        X_tr_sc = scaler.transform(X_tr)
        X_vl_sc = scaler.transform(X_vl)

        X_tr_seq, y_tr = build_sequences_chunked(X_tr_sc, tr_df["label"].values, lookback)
        X_vl_seq, y_vl = build_sequences_chunked(X_vl_sc, vl_df["label"].values, lookback)

        cw     = get_class_weights(pd.Series(y_tr))
        tr_ds  = build_tf_dataset(X_tr_seq, y_tr, batch_size=batch_size)
        vl_ds  = build_tf_dataset(X_vl_seq, y_vl, batch_size=batch_size)

        steps = max(len(X_tr_seq) // batch_size, 1)
        model = build_model(
            lookback=lookback, n_features=len(feature_cols),
            steps_per_epoch=steps, warmup_epochs=3, total_epochs=EPOCHS
        )
        ckpt = MODEL_DIR / f"wfv_fold{fold+1}.keras"
        model.fit(tr_ds, validation_data=vl_ds, epochs=EPOCHS,
                  class_weight=cw, callbacks=get_callbacks(ckpt), verbose=0)

        res    = model.evaluate(vl_ds, verbose=0)
        probs  = model.predict(vl_ds, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        dir_acc = (y_pred == y_vl).mean()

        results.append({
            "fold": fold+1,
            "val_loss": round(float(res[0]), 4),
            "val_accuracy": round(float(res[1]), 4),
            "directional_accuracy": round(float(dir_acc), 4),
            "train_rows": len(tr_df), "val_rows": len(vl_df),
        })
        logger.info(f"Fold {fold+1} | dir_acc={dir_acc:.4f}")
        del model
        keras.backend.clear_session()

    return results
