"""
====================================================================
DATA LOADER - Kaggle NSE Nifty 50 Minute Dataset
====================================================================
Loads the CSV, cleans it, filters market hours, and saves parquet.

Expected CSV columns: date, open, high, low, close, volume
Date format: "2015-01-01 09:15:00" or similar

Usage:
  Place nifty_50_minute.csv in data/ folder, then run:
  python data/loader.py
====================================================================
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    RAW_CSV_PATH, PROCESSED_PARQUET,
    CSV_DATE_COL, CSV_OPEN_COL, CSV_HIGH_COL,
    CSV_LOW_COL, CSV_CLOSE_COL, CSV_VOLUME_COL,
    LOG_LEVEL
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DataLoader")

# NSE market hours IST
MARKET_OPEN  = "09:15"
MARKET_CLOSE = "15:29"   # last 1-min bar starts at 15:29


def load_raw_csv(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load and do initial parse of the Kaggle CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Download from: kaggle.com/datasets/debashis74017/nifty-50-minute-data\n"
            f"Place nifty_50_minute.csv in: {path.parent}"
        )

    logger.info(f"Loading CSV: {path} ...")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Standardize column names
    2. Parse datetime index
    3. Filter to NSE market hours only
    4. Remove weekends and holidays (zero-volume days)
    5. Forward-fill tiny gaps (< 5 min)
    """

    # ── Rename columns ─────────────────────────────────────────
    col_map = {
        CSV_DATE_COL:   "datetime",
        CSV_OPEN_COL:   "open",
        CSV_HIGH_COL:   "high",
        CSV_LOW_COL:    "low",
        CSV_CLOSE_COL:  "close",
        CSV_VOLUME_COL: "volume",
    }
    # Case-insensitive rename
    df.columns = [c.strip().lower() for c in df.columns]
    col_map_lower = {k.lower(): v for k, v in col_map.items()}
    df = df.rename(columns=col_map_lower)

    missing = [c for c in ["open","high","low","close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}. "
                         f"Check CSV_*_COL settings in config.py")

    # ── Parse datetime ─────────────────────────────────────────
    df["datetime"] = pd.to_datetime(df["datetime"], infer_datetime_format=True)
    df = df.set_index("datetime")
    df.index = df.index.tz_localize("Asia/Kolkata", ambiguous="infer", nonexistent="shift_forward")
    df = df.sort_index()

    # ── Keep only OHLCV ────────────────────────────────────────
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.astype(float)

    # ── Drop bad rows ──────────────────────────────────────────
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]
    df = df[df["high"] >= df["low"]]

    # ── Filter market hours ────────────────────────────────────
    time_of_day = df.index.time
    import datetime as dt
    open_t  = dt.time(9, 15)
    close_t = dt.time(15, 29)
    df = df[(time_of_day >= open_t) & (time_of_day <= close_t)]

    # ── Remove weekends ────────────────────────────────────────
    df = df[df.index.dayofweek < 5]

    # ── Handle volume for index data (NIFTY 50 is an index, not a stock) ──
    if "volume" in df.columns:
        # For index data, volume is always 0 - this is normal
        # Only remove zero-volume bars if there's at least SOME non-zero volume
        non_zero_vol = (df["volume"] > 0).sum()
        if non_zero_vol > 0:
            # This is stock data with some volume
            zero_vol = (df["volume"] == 0).sum()
            if zero_vol > 0:
                logger.info(f"Removing {zero_vol} zero-volume bars")
                df = df[df["volume"] > 0]
        else:
            # This is index data - all volume is 0, which is expected
            logger.info("Index data detected (all volume = 0). This is normal for index data.")
    
    # ── Fill volume NaN with 0 ──
    df["volume"] = df["volume"].fillna(0)

    logger.info(
        f"Clean shape : {df.shape}\n"
        f"Date range  : {df.index[0]} -> {df.index[-1]}\n"
        f"Trading days: {df.index.normalize().nunique()}\n"
        f"NaN counts  :\n{df.isnull().sum()}"
    )
    return df


def validate_continuity(df: pd.DataFrame) -> None:
    """Check for large time gaps that indicate missing data."""
    gaps = df.index.to_series().diff()
    large_gaps = gaps[gaps > pd.Timedelta("10min")]
    intraday_gaps = large_gaps[
        large_gaps.index.time > pd.Timestamp("09:20").time()
    ]

    if len(intraday_gaps) > 0:
        logger.warning(
            f"Found {len(intraday_gaps)} intraday gaps > 10 min. "
            f"Largest: {large_gaps.max()}. "
            f"This is normal for circuit breaks and early market close days."
        )
    else:
        logger.info("Continuity check passed - no large intraday gaps")


def load_and_save(force: bool = False) -> pd.DataFrame:
    """
    Full pipeline: load CSV -> clean -> validate -> save parquet.
    Loads from parquet cache if already processed.
    """
    if PROCESSED_PARQUET.exists() and not force:
        logger.info(f"Loading cached parquet: {PROCESSED_PARQUET}")
        df = pd.read_parquet(PROCESSED_PARQUET)
        logger.info(f"Loaded {len(df):,} rows | {df.index[0]} -> {df.index[-1]}")
        return df

    raw = load_raw_csv()
    df  = clean_and_filter(raw)
    validate_continuity(df)

    df.to_parquet(PROCESSED_PARQUET)
    logger.info(f"Saved processed parquet: {PROCESSED_PARQUET}")

    # Print summary stats
    print(f"\n{'='*55}")
    print(f"DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Total rows     : {len(df):,}")
    print(f"  Trading days   : {df.index.normalize().nunique():,}")
    print(f"  Date range     : {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Years of data  : {(df.index[-1] - df.index[0]).days / 365:.1f}")
    print(f"  Avg rows/day   : {len(df) / df.index.normalize().nunique():.0f}")
    print(f"  Close range    : {df['close'].min():.0f} - {df['close'].max():.0f}")
    print(f"{'='*55}\n")

    return df


if __name__ == "__main__":
    df = load_and_save(force=True)
    print(df.tail(10))
