"""
====================================================================
MAIN RUNNER - RTX 3050 GPU Optimized
====================================================================
GPU setup happens at the very top — before ANY TensorFlow import.

Usage:
  # First verify GPU is detected:
  python main.py --mode gpu_check

  # Then run:
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode backtest
  python main.py --mode wfv
  python main.py --mode finetune --strategy partial
  python main.py --mode all
====================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# !! GPU SETUP FIRST — before any TF import !!
from utils.gpu_config import setup_gpu, check_cuda_installation, benchmark_gpu, get_optimal_batch_size
gpu_ok = setup_gpu(allow_growth=True, mixed_precision=True, verbose=True)

from config.config import LOG_DIR, MODEL_DIR, LOG_LEVEL, LOOKBACK, BATCH_SIZE

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "main.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("Main")


def run_train(force=False):
    from training.trainer import train
    return train(force_reload=force)


def run_evaluate(result=None):
    from tensorflow import keras
    from evaluation.evaluator import evaluate_model, CUSTOM_OBJECTS
    from data.loader import load_and_save
    from features.feature_engineering import build_features
    from features.label_generator import create_labels
    from features.sequence_builder import prepare_datasets

    model_path = MODEL_DIR / "nifty_kaggle_final.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run --mode train first.")

    model    = keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)
    raw      = load_and_save()
    featured = build_features(raw)
    labeled  = create_labels(featured)
    data     = prepare_datasets(labeled, fit_new_scaler=False)
    history  = result["history"] if result else None
    return evaluate_model(model, data, history=history)


def run_wfv():
    from data.loader import load_and_save
    from features.feature_engineering import build_features, get_feature_columns
    from features.label_generator import create_labels
    from training.trainer import walk_forward_validation

    raw      = load_and_save()
    featured = build_features(raw)
    labeled  = create_labels(featured)
    results  = walk_forward_validation(labeled, get_feature_columns())

    out = LOG_DIR / "wfv_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"WFV results: {out}")
    return results


def run_backtest():
    from tensorflow import keras
    import numpy as np
    from data.loader import load_and_save
    from features.feature_engineering import build_features
    from features.label_generator import create_labels
    from features.sequence_builder import prepare_datasets
    from evaluation.evaluator import CUSTOM_OBJECTS
    from config.config import CONFIDENCE_THRESHOLD

    model  = keras.models.load_model(
        str(MODEL_DIR / "nifty_kaggle_final.keras"), custom_objects=CUSTOM_OBJECTS
    )
    raw      = load_and_save()
    featured = build_features(raw)
    labeled  = create_labels(featured)
    data     = prepare_datasets(labeled, fit_new_scaler=False)

    y_probs  = model.predict(data["test_ds"], verbose=1)
    y_pred   = np.argmax(y_probs, axis=1)
    y_true   = data["y_test"]
    max_conf = y_probs.max(axis=1)
    mask     = max_conf >= CONFIDENCE_THRESHOLD

    test_df  = data["test_df"]
    fut_ret  = (test_df["future_close"].values / test_df["close"].values - 1)[LOOKBACK:]
    signal   = np.where((mask) & (y_pred == 0),  1,
               np.where((mask) & (y_pred == 1), -1, 0))
    pnl      = signal * fut_ret - (2e-4 * np.abs(signal))   # 2bps cost
    equity   = 1 + np.cumsum(pnl)
    dd       = (equity - np.maximum.accumulate(equity)) / (np.maximum.accumulate(equity) + 1e-9)

    tr_pnl   = pnl[signal != 0]
    n_trades = (signal != 0).sum()
    wins     = (tr_pnl > 0).sum()
    sharpe   = (tr_pnl.mean() / (tr_pnl.std() + 1e-9)) * (375 * 252) ** 0.5

    print(f"\n{'='*55}")
    print(f"  BACKTEST RESULTS (RTX 3050 model)")
    print(f"{'='*55}")
    print(f"  Confidence thresh : {CONFIDENCE_THRESHOLD}")
    print(f"  Trades taken      : {n_trades:,} ({n_trades/len(y_true)*100:.1f}%)")
    print(f"  Win Rate          : {wins/max(n_trades,1)*100:.1f}%")
    print(f"  Dir Acc (all)     : {(y_pred==y_true).mean()*100:.2f}%")
    print(f"  Dir Acc (filtered): {(y_pred[mask]==y_true[mask]).mean()*100:.2f}%")
    print(f"  Sharpe Ratio      : {sharpe:.4f}")
    print(f"  Max Drawdown      : {dd.min()*100:.2f}%")
    print(f"  Total PnL         : {pnl.sum():+.4f}")
    print(f"{'='*55}")

    # Save equity curve plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14,8), sharex=True,
                                       gridspec_kw={"height_ratios":[3,1]})
        a1.plot(equity, color="steelblue", lw=1.5)
        a1.set_title("Equity Curve - Kaggle Model (RTX 3050)", fontweight="bold")
        a1.set_ylabel("Portfolio Value"); a1.grid(True, alpha=0.3)
        a2.fill_between(range(len(dd)), dd, 0, alpha=0.5, color="orangered")
        a2.set_ylabel("Drawdown"); a2.set_xlabel("Timestep"); a2.grid(True, alpha=0.3)
        plt.tight_layout()
        p = LOG_DIR / "plots" / "equity_curve_kaggle.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        logger.info(f"Equity curve saved: {p}")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="NIFTY Kaggle Predictor (RTX 3050)")
    parser.add_argument("--mode", default="train",
                        choices=["gpu_check", "train", "evaluate", "backtest",
                                 "wfv", "finetune", "all"])
    parser.add_argument("--force_reload", action="store_true",
                        help="Re-process CSV even if parquet cache exists")
    parser.add_argument("--strategy", default="partial",
                        choices=["partial", "full", "head_only"])
    parser.add_argument("--recent_rows", type=int, default=50000)
    args = parser.parse_args()

    batch_size = get_optimal_batch_size("transformer") if gpu_ok else BATCH_SIZE

    print(f"\n{'#'*60}")
    print(f"  NIFTY 50 KAGGLE PREDICTOR")
    print(f"  GPU      : {'RTX 3050 (ACTIVE)' if gpu_ok else 'CPU ONLY (install CUDA!)'}")
    print(f"  Mode     : {args.mode.upper()}")
    print(f"  Lookback : {LOOKBACK} x 1min")
    print(f"  Batch    : {batch_size}")
    print(f"{'#'*60}\n")

    if args.mode == "gpu_check":
        check_cuda_installation()
        benchmark_gpu()
        return

    result = None
    if args.mode in ("train", "all"):
        result = run_train(force=args.force_reload)

    if args.mode in ("evaluate", "all"):
        run_evaluate(result)

    if args.mode in ("wfv", "all"):
        run_wfv()

    if args.mode in ("backtest", "all"):
        run_backtest()

    if args.mode == "finetune":
        from finetuning.finetuner import finetune
        finetune(strategy=args.strategy, recent_rows=args.recent_rows)


if __name__ == "__main__":
    main()
