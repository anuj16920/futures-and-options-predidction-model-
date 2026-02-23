"""
====================================================================
NIFTY 50 KAGGLE DATASET - MASTER CONFIG
====================================================================
Dataset : NSE Nifty 50 Index Minute Data (2015-2026)
Source  : kaggle.com/datasets/debashis74017/nifty-50-minute-data
Rows    : ~945,000 (1-min candles, 10 years)
File    : nifty_50_minute.csv  (date, open, high, low, close, volume)

With this much data we can:
  - Use LOOKBACK=60 (60 min = 1 hour context)
  - Deep architecture (Transformer + CNN + LSTM)
  - Multi-timeframe features (1m + 5m + 15m aggregated)
  - ~500K parameter model without overfitting
====================================================================
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
SCALER_DIR = BASE_DIR / "scalers"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, SCALER_DIR, LOG_DIR, LOG_DIR / "plots"]:
    d.mkdir(parents=True, exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────
# Place your downloaded CSV here:
RAW_CSV_PATH = DATA_DIR / "nifty_50_minute.csv"
PROCESSED_PARQUET = DATA_DIR / "nifty_processed.parquet"

# Column name mapping from Kaggle CSV
# Adjust if your CSV has different column names
CSV_DATE_COL   = "date"
CSV_OPEN_COL   = "open"
CSV_HIGH_COL   = "high"
CSV_LOW_COL    = "low"
CSV_CLOSE_COL  = "close"
CSV_VOLUME_COL = "volume"

TIMEFRAME = "1m"      # Base timeframe of the dataset

# ── SEQUENCE / PREDICTION ─────────────────────────────────────────
LOOKBACK = 60         # 60 x 1min = 60 min (1 hour) of context
HORIZON  = 5          # Predict 5 candles = 5 minutes ahead
NUM_CLASSES = 3       # Bullish=0, Bearish=1, Neutral=2

# ── LABEL THRESHOLDS ──────────────────────────────────────────────
# With large data, k=0.4 gives good balance
# Expected: ~33% Bull, ~33% Bear, ~34% Neutral
ATR_MULTIPLIER_K = 0.4

# ── MULTI-TIMEFRAME AGGREGATION ───────────────────────────────────
# We build features at 3 timeframes and stack them
MTF_WINDOWS = [1, 5, 15]   # minutes (1m, 5m, 15m)

# ── TECHNICAL INDICATORS ──────────────────────────────────────────
RSI_PERIOD       = 14
EMA_SHORT        = 9
EMA_MEDIUM       = 21
EMA_LONG         = 50
ATR_PERIOD       = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD    = 2.0
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
VOLATILITY_WIN   = 20
MOMENTUM_SHIFT   = 5
VWAP_RESET_DAILY = True

# ── MODEL ARCHITECTURE ────────────────────────────────────────────
# Transformer-CNN-LSTM hybrid
# ~945K rows -> can support ~450K params comfortably
TRANSFORMER_HEADS    = 4
TRANSFORMER_DIM      = 64     # must be divisible by TRANSFORMER_HEADS
TRANSFORMER_FF_DIM   = 128    # feedforward dim inside transformer block
TRANSFORMER_BLOCKS   = 2      # number of transformer encoder blocks
TRANSFORMER_DROPOUT  = 0.1

CNN_FILTERS          = 64
CNN_KERNEL_SIZE      = 3

LSTM_UNITS           = 128
LSTM_LAYERS          = 2

DENSE_UNITS          = 128
DROPOUT_RATE         = 0.3
L2_REG               = 1e-4

# ── TRAINING ──────────────────────────────────────────────────────
EPOCHS               = 100
BATCH_SIZE           = 512    # Large batch for large dataset
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-4
WARMUP_EPOCHS        = 5      # Cosine LR warmup
LABEL_SMOOTHING      = 0.05
FOCAL_GAMMA          = 2.0

EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE      = 7
LR_REDUCE_FACTOR        = 0.5
LR_MIN                  = 1e-6

# ── DATA SPLIT (CHRONOLOGICAL) ────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── WALK-FORWARD VALIDATION ───────────────────────────────────────
WFV_N_SPLITS     = 5
WFV_TRAIN_WINDOW = 0.60
WFV_STEP_SIZE    = 0.08

# ── BACKTEST ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
TRADE_COST_BPS       = 2

# ── FINE-TUNING ───────────────────────────────────────────────────
FINETUNE_EPOCHS          = 20
FINETUNE_LR              = 5e-5
FINETUNE_RECENT_ROWS     = 50000   # Last ~50K rows for fine-tuning
FINETUNE_LAYERS_UNFREEZE = ["transformer", "lstm", "dense"]
FINETUNE_BATCH_SIZE      = 256

# ── LOGGING ───────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
