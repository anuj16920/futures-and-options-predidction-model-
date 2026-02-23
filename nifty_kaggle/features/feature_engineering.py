"""
====================================================================
FEATURE ENGINEERING - Multi-Timeframe (1m + 5m + 15m)
====================================================================
With 945K rows we can afford rich feature sets:

  BASE (1m):
    RSI, EMA9/21/50, ATR, VWAP, Returns, Volatility,
    Bollinger Bands, MACD, Volume features, Time encoding,
    HL Range, CO Spread, Momentum, Market session flags

  AGGREGATED (5m, 15m):
    Higher-timeframe OHLCV resampled, then same indicators
    computed on the resampled data and merged back to 1m index.
    This gives the model "zoom-out" context.

  MARKET MICROSTRUCTURE:
    Bid-ask spread proxy, volume imbalance, tick direction,
    intraday cumulative return, distance from day H/L

Total features: ~35
No lookahead bias: all rolling/ewm use past data only.
====================================================================
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    RSI_PERIOD, EMA_SHORT, EMA_MEDIUM, EMA_LONG,
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    VOLATILITY_WIN, MOMENTUM_SHIFT, MTF_WINDOWS, LOG_LEVEL
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("FeatureEngineering")


# ─────────────────────────────────────────────────────────────────
# BASE INDICATORS
# ─────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return (100 - 100 / (1 + gain / (loss + 1e-9))).rename("rsi")


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean().rename(f"ema_{span}")


def atr(high, low, close, period: int = ATR_PERIOD) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean().rename("atr")


def bollinger(close: pd.Series, period: int = BOLLINGER_PERIOD, std: float = BOLLINGER_STD):
    mid  = close.rolling(period, min_periods=period//2).mean()
    band = close.rolling(period, min_periods=period//2).std()
    upper = mid + std * band
    lower = mid - std * band
    pct_b = (close - lower) / (upper - lower + 1e-9)   # 0=lower, 1=upper
    width = (upper - lower) / (mid + 1e-9)              # normalized bandwidth
    return pct_b.rename("bb_pct"), width.rename("bb_width")


def macd(close: pd.Series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    fast_ema  = close.ewm(span=fast,   adjust=False).mean()
    slow_ema  = close.ewm(span=slow,   adjust=False).mean()
    macd_line = fast_ema - slow_ema
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line
    # Normalize by close price for scale invariance
    return (macd_line / close).rename("macd_norm"), (histogram / close).rename("macd_hist_norm")


def vwap_daily(high, low, close, volume) -> pd.Series:
    """Intraday VWAP reset at market open each day."""
    typical = (high + low + close) / 3
    
    # For index data with zero volume, use simple daily average price
    if (volume == 0).all():
        df_tmp = pd.DataFrame({"tp": typical, "date": close.index.date}, index=close.index)
        vals = np.empty(len(df_tmp))
        for date, grp in df_tmp.groupby("date"):
            idx = [df_tmp.index.get_loc(i) for i in grp.index]
            vals[idx] = grp["tp"].expanding().mean().values
        return pd.Series(vals, index=close.index, name="vwap")
    
    # For stock data with volume
    tv      = typical * volume
    df_tmp  = pd.DataFrame({"tv": tv, "vol": volume, "date": close.index.date}, index=close.index)
    vals    = np.empty(len(df_tmp))
    for date, grp in df_tmp.groupby("date"):
        idx  = [df_tmp.index.get_loc(i) for i in grp.index]
        c_tv = grp["tv"].cumsum().values
        c_v  = grp["vol"].cumsum().values
        vals[idx] = c_tv / (c_v + 1e-9)
    return pd.Series(vals, index=close.index, name="vwap")


def volume_features(volume: pd.Series, window: int = 20):
    # For index data where volume is always 0, return constant features
    if (volume == 0).all():
        ratio = pd.Series(1.0, index=volume.index, name="vol_ratio")
        vol_trend = pd.Series(0.0, index=volume.index, name="vol_trend")
        return ratio, vol_trend
    
    avg  = volume.rolling(window, min_periods=5).mean()
    ratio = (volume / (avg + 1e-9)).rename("vol_ratio")
    # Volume trend: is current bar volume increasing?
    vol_trend = volume.diff().apply(np.sign).rename("vol_trend")
    return ratio, vol_trend


def time_encoding(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sin/cos encoding for minute-of-day and day-of-week."""
    mof = index.hour * 60 + index.minute
    OPEN_MIN, DUR = 9*60+15, 375
    norm_mof = (mof - OPEN_MIN) / DUR * 2 * np.pi
    dow = index.dayofweek / 4 * 2 * np.pi    # 0=Mon, 4=Fri

    return pd.DataFrame({
        "time_sin"  : np.sin(norm_mof),
        "time_cos"  : np.cos(norm_mof),
        "dow_sin"   : np.sin(dow),
        "dow_cos"   : np.cos(dow),
    }, index=index)


def market_session_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Flag the first 30min and last 30min of session — high volatility zones."""
    mof = index.hour * 60 + index.minute
    return pd.DataFrame({
        "session_open"  : ((mof >= 555) & (mof <= 585)).astype(float),   # 9:15-9:45
        "session_close" : ((mof >= 900) & (mof <= 929)).astype(float),   # 15:00-15:29
    }, index=index)


def intraday_context(high, low, close) -> pd.DataFrame:
    """
    Cumulative intraday return and distance from day's H/L.
    Resets each day. No lookahead.
    """
    df_tmp = pd.DataFrame({"h": high, "l": low, "c": close,
                           "date": close.index.date}, index=close.index)
    cum_ret = np.zeros(len(df_tmp))
    dist_h  = np.zeros(len(df_tmp))
    dist_l  = np.zeros(len(df_tmp))

    for date, grp in df_tmp.groupby("date"):
        idx     = [df_tmp.index.get_loc(i) for i in grp.index]
        open_c  = grp["c"].iloc[0]
        day_h   = grp["h"].cummax().values
        day_l   = grp["l"].cummin().values
        c_vals  = grp["c"].values
        cum_ret[idx] = (c_vals - open_c) / (open_c + 1e-9)
        dist_h[idx]  = (day_h - c_vals) / (c_vals + 1e-9)
        dist_l[idx]  = (c_vals - day_l) / (c_vals + 1e-9)

    return pd.DataFrame({
        "intraday_return": cum_ret,
        "dist_from_high" : dist_h,
        "dist_from_low"  : dist_l,
    }, index=close.index)


# ─────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME FEATURES
# ─────────────────────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1m OHLCV to N-minute bars."""
    rule = f"{minutes}T"
    resampled = df.resample(rule, closed="right", label="right").agg({
        "open"  : "first",
        "high"  : "max",
        "low"   : "min",
        "close" : "last",
        "volume": "sum",
    }).dropna(subset=["close"])
    return resampled


def build_mtf_features(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Compute features on N-minute bars, then forward-fill back to 1m index.
    Uses .reindex + ffill so no future data leaks into past bars.
    """
    df_nm = resample_ohlcv(df_1m, minutes)

    feat = pd.DataFrame(index=df_nm.index)
    feat[f"rsi_{minutes}m"]      = rsi(df_nm["close"])
    feat[f"ema9_{minutes}m_norm"]  = (ema(df_nm["close"], 9) - df_nm["close"]) / df_nm["close"]
    feat[f"ema21_{minutes}m_norm"] = (ema(df_nm["close"], 21) - df_nm["close"]) / df_nm["close"]
    atr_nm = atr(df_nm["high"], df_nm["low"], df_nm["close"])
    feat[f"atr_{minutes}m_norm"]   = atr_nm / df_nm["close"]
    bb_pct, bb_w = bollinger(df_nm["close"])
    feat[f"bb_pct_{minutes}m"]  = bb_pct
    feat[f"bb_w_{minutes}m"]    = bb_w
    macd_n, macd_h = macd(df_nm["close"])
    feat[f"macd_{minutes}m"]    = macd_n
    feat[f"macdh_{minutes}m"]   = macd_h
    feat[f"ret_{minutes}m"]     = df_nm["close"].pct_change()

    # Forward-fill onto 1m index (past N-min bar value valid until next bar)
    feat_1m = feat.reindex(df_1m.index, method="ffill")
    return feat_1m


# ─────────────────────────────────────────────────────────────────
# MASTER FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features on the 1m Kaggle dataframe.

    Returns DataFrame with all feature columns.
    NaN warmup rows are dropped.
    """
    logger.info(f"Building features on {len(df):,} rows...")
    feat = df.copy()

    c, h, l, o, v = feat["close"], feat["high"], feat["low"], feat["open"], feat["volume"]

    # ── 1m Base Indicators ─────────────────────────────────────
    feat["rsi"]         = rsi(c)
    e9  = ema(c, EMA_SHORT)
    e21 = ema(c, EMA_MEDIUM)
    e50 = ema(c, EMA_LONG)
    feat["ema9_norm"]   = (e9  - c) / c
    feat["ema21_norm"]  = (e21 - c) / c
    feat["ema50_norm"]  = (e50 - c) / c
    feat["ema9_21_cross"]  = (e9  - e21) / c
    feat["ema21_50_cross"] = (e21 - e50) / c

    atr_1m = atr(h, l, c)
    feat["atr"]         = atr_1m  # Raw ATR for label generation
    feat["atr_norm"]    = atr_1m / c

    feat["vwap"]        = vwap_daily(h, l, c, v)
    feat["vwap_dev"]    = (c - feat["vwap"]) / feat["vwap"]

    feat["returns"]     = np.log(c / c.shift(1))
    feat["volatility"]  = feat["returns"].rolling(VOLATILITY_WIN, min_periods=5).std()

    feat["hl_range"]    = (h - l) / c
    feat["co_spread"]   = (c - o) / c
    feat[f"mom_{MOMENTUM_SHIFT}"] = (c - c.shift(MOMENTUM_SHIFT)) / c

    bb_pct, bb_w = bollinger(c)
    feat["bb_pct"]  = bb_pct
    feat["bb_width"] = bb_w

    macd_n, macd_h = macd(c)
    feat["macd_norm"]      = macd_n
    feat["macd_hist_norm"] = macd_h

    vol_r, vol_t = volume_features(v)
    feat["vol_ratio"] = vol_r
    feat["vol_trend"] = vol_t

    # ── Time Encoding ──────────────────────────────────────────
    te = time_encoding(feat.index)
    for col in te.columns:
        feat[col] = te[col]

    # ── Session Flags ──────────────────────────────────────────
    sf = market_session_flags(feat.index)
    for col in sf.columns:
        feat[col] = sf[col]

    # ── Intraday Context ───────────────────────────────────────
    ic = intraday_context(h, l, c)
    for col in ic.columns:
        feat[col] = ic[col]

    # ── Multi-Timeframe Features (5m, 15m) ────────────────────
    for tf_min in [m for m in MTF_WINDOWS if m > 1]:
        logger.info(f"  Building {tf_min}m features...")
        mtf = build_mtf_features(df, tf_min)
        for col in mtf.columns:
            feat[col] = mtf[col]

    # ── Drop warmup NaN rows ───────────────────────────────────
    before = len(feat)
    # First, fill any remaining inf values
    feat = feat.replace([np.inf, -np.inf], np.nan)
    
    # Check which columns have NaN
    nan_counts = feat.isnull().sum()
    if nan_counts.sum() > 0:
        logger.info(f"NaN counts before dropping:\n{nan_counts[nan_counts > 0]}")
    
    # Drop rows with ANY NaN
    feat = feat.dropna()
    
    if len(feat) == 0:
        raise ValueError(
            f"All rows were dropped after feature engineering! "
            f"This usually means the multi-timeframe resampling failed. "
            f"Original rows: {before}"
        )
    
    logger.info(
        f"Features built. Dropped {before - len(feat):,} warmup rows. "
        f"Final: {feat.shape}"
    )
    return feat


def get_feature_columns() -> list:
    """
    Canonical ordered list of features used as model input.
    Must be consistent across training, inference, fine-tuning.
    """
    return [
        # 1m Base
        "rsi",
        "ema9_norm", "ema21_norm", "ema50_norm",
        "ema9_21_cross", "ema21_50_cross",
        "atr_norm",
        "vwap_dev",
        "returns", "volatility",
        "hl_range", "co_spread", f"mom_{MOMENTUM_SHIFT}",
        "bb_pct", "bb_width",
        "macd_norm", "macd_hist_norm",
        "vol_ratio", "vol_trend",
        # Time
        "time_sin", "time_cos", "dow_sin", "dow_cos",
        # Session
        "session_open", "session_close",
        # Intraday
        "intraday_return", "dist_from_high", "dist_from_low",
        # 5m MTF
        "rsi_5m", "ema9_5m_norm", "ema21_5m_norm", "atr_5m_norm",
        "bb_pct_5m", "bb_w_5m", "macd_5m", "macdh_5m", "ret_5m",
        # 15m MTF
        "rsi_15m", "ema9_15m_norm", "ema21_15m_norm", "atr_15m_norm",
        "bb_pct_15m", "bb_w_15m", "macd_15m", "macdh_15m", "ret_15m",
    ]


if __name__ == "__main__":
    from data.loader import load_and_save
    df = load_and_save()
    features = build_features(df)
    cols = get_feature_columns()
    print(f"\nFeatures ({len(cols)}): {cols}")
    print(f"\nSample:\n{features[cols].tail(3)}")
    print(f"\nNaN check:\n{features[cols].isnull().sum()}")
