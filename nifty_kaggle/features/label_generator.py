"""
====================================================================
LABEL GENERATOR - ATR-Adjusted Dynamic Thresholds
====================================================================
Same leakage-free approach but tuned for 1m data:
  future_move = close[t + HORIZON] - close[t]
  Bullish  : future_move >  k * ATR[t]
  Bearish  : future_move < -k * ATR[t]
  Neutral  : otherwise

With 945K rows and k=0.4 on 1m ATR expect:
  ~33% Bullish / ~33% Bearish / ~34% Neutral (very balanced)
====================================================================
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import HORIZON, ATR_MULTIPLIER_K, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("LabelGenerator")

LABEL_MAP   = {"Bullish": 0, "Bearish": 1, "Neutral": 2}
LABEL_NAMES = {0: "Bullish", 1: "Bearish", 2: "Neutral"}


def create_labels(df: pd.DataFrame, horizon: int = HORIZON,
                  k: float = ATR_MULTIPLIER_K) -> pd.DataFrame:
    """
    Generate ATR-adjusted directional labels. No lookahead.

    Returns df with: future_close, future_move, threshold, label, label_name
    Last `horizon` rows dropped (no future available).
    """
    logger.info(f"Creating labels | horizon={horizon} | k={k}")
    df = df.copy()

    df["future_close"] = df["close"].shift(-horizon)
    df["future_move"]  = df["future_close"] - df["close"]
    df["threshold"]    = k * df["atr"]

    conditions = [
        df["future_move"] >  df["threshold"],
        df["future_move"] < -df["threshold"],
    ]
    df["label"]      = np.select(conditions, [0, 1], default=2)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df = df.dropna(subset=["future_close"])

    dist = df["label_name"].value_counts()
    total = len(df)
    logger.info(
        f"Label distribution (total={total:,}):\n" +
        "\n".join(f"  {n:<10}: {c:>7,} ({c/total*100:.1f}%)" for n, c in dist.items())
    )
    return df


def get_class_weights(labels: pd.Series) -> dict:
    """Inverse-frequency class weights for Keras class_weight arg."""
    counts  = labels.value_counts().sort_index()
    total   = counts.sum()
    n_cls   = len(counts)
    weights = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
    logger.info("Class weights: " + str({LABEL_NAMES[k]: round(v, 4) for k, v in weights.items()}))
    return weights


def validate_labels(df: pd.DataFrame) -> None:
    assert df["label"].isnull().sum() == 0, "NaN labels!"
    assert df["future_close"].isnull().sum() == 0, "NaN future_close!"
    bull = df[df["label"] == 0]
    bear = df[df["label"] == 1]
    assert (bull["future_move"] > 0).all(), "Bullish rows with negative future_move!"
    assert (bear["future_move"] < 0).all(), "Bearish rows with positive future_move!"
    logger.info("Label validation passed")
