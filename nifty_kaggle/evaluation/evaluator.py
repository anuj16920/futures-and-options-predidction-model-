"""
====================================================================
EVALUATION - Full metrics suite for Kaggle model
====================================================================
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
)
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import CONFIDENCE_THRESHOLD, LOG_DIR, LOG_LEVEL, NUM_CLASSES
from features.label_generator import LABEL_NAMES
from models.architecture import FocalLoss, WarmupCosineSchedule, PositionalEncoding, TransformerEncoderBlock

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("Evaluator")
PLOT_DIR = LOG_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["Bullish", "Bearish", "Neutral"]

CUSTOM_OBJECTS = {
    "FocalLoss": FocalLoss,
    "WarmupCosineSchedule": WarmupCosineSchedule,
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}


def compute_metrics(y_true, y_pred, y_probs, split="Test"):
    acc       = accuracy_score(y_true, y_pred)
    f1_mac    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_mac  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac   = recall_score(y_true, y_pred, average="macro",     zero_division=0)
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per   = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per    = f1_score(y_true, y_pred, average=None, zero_division=0)

    try:
        y_oh  = np.eye(NUM_CLASSES)[y_true.astype(int)]
        auc   = roc_auc_score(y_oh, y_probs, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    max_conf     = y_probs.max(axis=1)
    hc_mask      = max_conf >= CONFIDENCE_THRESHOLD
    hc_acc       = accuracy_score(y_true[hc_mask], y_pred[hc_mask]) if hc_mask.sum() > 0 else float("nan")

    print(f"\n{'='*65}")
    print(f"  {split.upper()} EVALUATION  ({len(y_true):,} samples)")
    print(f"{'='*65}")
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1  (macro)      : {f1_mac:.4f}")
    print(f"  F1  (weighted)   : {f1_wt:.4f}")
    print(f"  Precision (macro): {prec_mac:.4f}")
    print(f"  Recall    (macro): {rec_mac:.4f}")
    print(f"  AUC-ROC   (macro): {auc:.4f}")
    print(f"  High-conf (>={CONFIDENCE_THRESHOLD}): {hc_mask.sum():,} ({hc_mask.mean()*100:.1f}%) | Acc: {hc_acc:.4f}")
    print(f"\n  {'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>10}")
    print(f"  {'-'*48}")
    for i in range(min(NUM_CLASSES, len(f1_per))):
        sup = int((y_true == i).sum())
        print(f"  {CLASS_NAMES[i]:<12} {prec_per[i]:>8.4f} {rec_per[i]:>8.4f} {f1_per[i]:>8.4f} {sup:>10,}")
    print(f"{'='*65}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    return {
        "split": split, "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4), "f1_weighted": round(f1_wt, 4),
        "precision_macro": round(prec_mac, 4), "recall_macro": round(rec_mac, 4),
        "auc_roc": round(auc, 4),
        "high_conf_count": int(hc_mask.sum()),
        "high_conf_accuracy": round(float(hc_acc), 4) if not np.isnan(hc_acc) else None,
        "per_class": {CLASS_NAMES[i]: {
            "precision": round(float(prec_per[i]), 4),
            "recall": round(float(rec_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int((y_true == i).sum()),
        } for i in range(min(NUM_CLASSES, len(f1_per)))}
    }


def plot_confusion_matrix(y_true, y_pred, title="Test", save=True):
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, sfx in zip(axes, [cm_norm, cm], [".2f","d"], ["Normalized","Raw"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, linewidths=0.5)
        ax.set_title(f"{title} Confusion Matrix ({sfx})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    if save:
        p = PLOT_DIR / f"cm_{title.lower().replace(' ','_')}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_history(history, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    pairs = [("loss","val_loss","Loss"), ("accuracy","val_accuracy","Accuracy"),
             ("precision","val_precision","Precision"), ("recall","val_recall","Recall")]
    for ax, (tr_k, vl_k, title) in zip(axes.flatten(), pairs):
        if tr_k in history:
            ax.plot(history[tr_k], label="Train", color="steelblue")
        if vl_k in history:
            ax.plot(history[vl_k], label="Val", color="orangered")
        ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(model, data, history=None):
    all_metrics = {}
    for split, X, y in [("Train", data["X_train"], data["y_train"]),
                         ("Val",   data["X_val"],   data["y_val"]),
                         ("Test",  data["X_test"],  data["y_test"])]:
        probs = model.predict(data[f"{split.lower()}_ds"], verbose=0)
        pred  = np.argmax(probs, axis=1)
        m     = compute_metrics(y, pred, probs, split)
        all_metrics[split.lower()] = m
        plot_confusion_matrix(y, pred, title=split)

    if history:
        plot_history(history)

    out = LOG_DIR / "eval_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {out}")
    return all_metrics
