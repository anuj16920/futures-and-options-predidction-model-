"""
====================================================================
FINE-TUNING - Kaggle Model Incremental Updates
====================================================================
Use weekly to incorporate latest market data into the model.
Supports partial/full/head-only strategies.
====================================================================
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_RECENT_ROWS,
    FINETUNE_LAYERS_UNFREEZE, FINETUNE_BATCH_SIZE,
    MODEL_DIR, LOG_DIR, LOG_LEVEL, LOOKBACK
)
from features.feature_engineering import build_features, get_feature_columns
from features.label_generator import create_labels, get_class_weights, validate_labels
from features.sequence_builder import build_sequences_chunked, load_scaler, build_tf_dataset
from models.architecture import (
    get_callbacks, FocalLoss, WarmupCosineSchedule,
    PositionalEncoding, TransformerEncoderBlock
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(LOG_DIR / "finetune.log", encoding="utf-8")
                    ])
logger = logging.getLogger("FineTuner")

CUSTOM_OBJECTS = {
    "FocalLoss": FocalLoss,
    "WarmupCosineSchedule": WarmupCosineSchedule,
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}


def load_model(path=None):
    if path is None:
        path = MODEL_DIR / "nifty_kaggle_final.keras"
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}. Run training first.")
    model = keras.models.load_model(str(path), custom_objects=CUSTOM_OBJECTS)
    logger.info(f"Loaded model: {path} | params={model.count_params():,}")
    return model


def set_layer_trainability(model, strategy="partial"):
    """
    partial   : Freeze CNN, unfreeze Transformer + LSTM + Dense
    full      : Unfreeze all layers
    head_only : Only Dense output head
    """
    if strategy == "full":
        for layer in model.layers:
            layer.trainable = True
    elif strategy == "head_only":
        for layer in model.layers:
            layer.trainable = any(k in layer.name.lower() for k in ["dense", "output"])
    else:  # partial
        freeze_keywords = ["cnn", "pos_enc", "input_proj", "bn_cnn"]
        for layer in model.layers:
            layer.trainable = not any(k in layer.name.lower() for k in freeze_keywords)

    trainable = sum(1 for l in model.layers if l.trainable)
    total     = len(model.layers)
    tp = sum(__import__("tensorflow").size(w).numpy() for w in model.trainable_weights)
    logger.info(f"Strategy={strategy} | Trainable layers: {trainable}/{total} | Params: {tp:,}")


def finetune(
    strategy="partial",
    recent_rows=FINETUNE_RECENT_ROWS,
    epochs=FINETUNE_EPOCHS,
    lr=FINETUNE_LR,
    batch_size=FINETUNE_BATCH_SIZE,
    base_model_path=None,
    save=True,
    data_df=None,
):
    """
    Fine-tune the Kaggle model on recent data.

    Parameters
    ----------
    strategy     : 'partial' | 'full' | 'head_only'
    recent_rows  : How many most-recent rows to use (default 50K = ~33 days)
    epochs       : Max fine-tuning epochs
    lr           : Fine-tuning learning rate (keep small: 1e-5 to 1e-4)
    batch_size   : Batch size
    base_model_path : Path to base model (None = canonical)
    save         : Whether to save the fine-tuned model
    data_df      : Pre-loaded labeled DataFrame (None = reload from disk)
    """
    logger.info(f"Fine-tuning | strategy={strategy} | recent_rows={recent_rows:,} | lr={lr}")

    model = load_model(base_model_path)
    set_layer_trainability(model, strategy)

    # Recompile with small LR
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5, clipnorm=1.0),
        loss=FocalLoss(),
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )

    # ── Prepare data ──────────────────────────────────────────
    if data_df is None:
        from data.loader import load_and_save
        raw = load_and_save()
        featured = build_features(raw)
        labeled  = create_labels(featured)
    else:
        labeled = data_df

    validate_labels(labeled)

    # Use only the most recent N rows
    if len(labeled) > recent_rows:
        labeled = labeled.iloc[-recent_rows:].copy()
        logger.info(f"Using last {recent_rows:,} rows: {labeled.index[0].date()} -> {labeled.index[-1].date()}")

    # Chronological 80/20 split
    n         = len(labeled)
    val_start = int(n * 0.8)
    tr_df     = labeled.iloc[:val_start]
    vl_df     = labeled.iloc[val_start:]
    feat_cols = get_feature_columns()

    scaler    = load_scaler()
    X_tr_sc   = scaler.transform(tr_df[feat_cols].values.astype("float32"))
    X_vl_sc   = scaler.transform(vl_df[feat_cols].values.astype("float32"))

    X_tr, y_tr = build_sequences_chunked(X_tr_sc, tr_df["label"].values, LOOKBACK)
    X_vl, y_vl = build_sequences_chunked(X_vl_sc, vl_df["label"].values, LOOKBACK)

    cw     = get_class_weights(pd.Series(y_tr))
    tr_ds  = build_tf_dataset(X_tr, y_tr, batch_size=batch_size)
    vl_ds  = build_tf_dataset(X_vl, y_vl, batch_size=batch_size)

    # ── Train ─────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt  = MODEL_DIR / f"finetune_{strategy}_{ts}.keras"
    cbs   = get_callbacks(ckpt)

    history = model.fit(
        tr_ds, validation_data=vl_ds,
        epochs=epochs, class_weight=cw,
        callbacks=cbs, verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────
    from sklearn.metrics import f1_score, accuracy_score
    probs   = model.predict(vl_ds, verbose=0)
    y_pred  = np.argmax(probs, axis=1)
    val_acc = accuracy_score(y_vl, y_pred)
    val_f1  = f1_score(y_vl, y_pred, average="macro", zero_division=0)

    logger.info(f"Fine-tune done | val_acc={val_acc:.4f} | f1_macro={val_f1:.4f}")

    if save:
        prod_path = MODEL_DIR / "nifty_kaggle_final.keras"
        ans = input(f"\nOverwrite production model? val_acc={val_acc:.4f} f1={val_f1:.4f} [y/N]: ")
        if ans.strip().lower() == "y":
            model.save(str(prod_path))
            logger.info(f"Production model updated: {prod_path}")

    # Log
    log_entry = {
        "timestamp": ts, "strategy": strategy, "recent_rows": recent_rows,
        "lr": lr, "epochs": len(history.history["loss"]),
        "val_accuracy": round(val_acc, 4), "val_f1_macro": round(val_f1, 4),
        "checkpoint": str(ckpt)
    }
    log_path = LOG_DIR / "finetune_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"model": model, "metrics": log_entry}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="partial", choices=["partial","full","head_only"])
    p.add_argument("--recent_rows", type=int, default=FINETUNE_RECENT_ROWS)
    p.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
    p.add_argument("--lr", type=float, default=FINETUNE_LR)
    args = p.parse_args()
    finetune(strategy=args.strategy, recent_rows=args.recent_rows,
             epochs=args.epochs, lr=args.lr)
