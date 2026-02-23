"""
====================================================================
GPU CONFIGURATION - NVIDIA RTX 3050
====================================================================
RTX 3050 Specs:
  VRAM        : 4GB (laptop) or 8GB (desktop)
  CUDA Cores  : 2560
  Architecture: Ampere (sm_86)
  TF32        : Supported (faster matmul)
  Mixed Prec  : FP16 supported

This module must be imported FIRST before any TensorFlow code.
Call setup_gpu() at the very start of your training script.
====================================================================
"""

import os
import sys
import logging

logger = logging.getLogger("GPUConfig")


def setup_gpu(
    memory_limit_mb: int = None,
    allow_growth: bool = True,
    mixed_precision: bool = True,
    verbose: bool = True
) -> bool:
    """
    Configure TensorFlow for RTX 3050.

    Parameters
    ----------
    memory_limit_mb : Hard cap on VRAM in MB.
                      None = use all available (recommended for RTX 3050 4GB)
                      Set to 3500 if you get OOM errors.
    allow_growth    : Allocate VRAM incrementally (prevents hogging all 4GB upfront)
    mixed_precision : Use FP16 for compute, FP32 for weights.
                      ~2x speedup on Ampere GPUs. Highly recommended.
    verbose         : Print GPU info

    Returns
    -------
    bool : True if GPU configured, False if CPU fallback
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logger.warning(
            "No GPU detected! Running on CPU.\n"
            "Fix: Install CUDA 11.8 + cuDNN 8.6:\n"
            "  https://developer.nvidia.com/cuda-11-8-0-download-archive\n"
            "  https://developer.nvidia.com/rdp/cudnn-archive"
        )
        return False

    try:
        for gpu in gpus:
            if allow_growth:
                tf.config.experimental.set_memory_growth(gpu, True)

            if memory_limit_mb is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )]
                )

        # ── Mixed Precision (FP16) ─────────────────────────────
        # RTX 3050 Ampere supports FP16 tensor cores
        # ~1.5-2x training speedup with minimal accuracy impact
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision: ON (float16 compute, float32 weights)")

        # ── TF32 for matrix ops ────────────────────────────────
        # Ampere-specific: faster than FP32, more accurate than FP16
        tf.config.experimental.enable_tensor_float_32_execution(True)

        # ── XLA JIT compilation ────────────────────────────────
        # Fuses ops for faster GPU execution
        tf.config.optimizer.set_jit(True)

        if verbose:
            _print_gpu_info(tf, gpus, mixed_precision, memory_limit_mb)

        return True

    except RuntimeError as e:
        logger.error(f"GPU config failed: {e}")
        return False


def _print_gpu_info(tf, gpus, mixed_precision, memory_limit_mb):
    """Print GPU configuration summary."""
    print(f"\n{'='*55}")
    print(f"  GPU CONFIGURATION")
    print(f"{'='*55}")
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        name    = details.get("device_name", "Unknown GPU")
        cc      = details.get("compute_capability", "?")
        print(f"  GPU {i}      : {name}")
        print(f"  Compute Cap : {cc}")

    print(f"  Mixed Prec  : {'FP16 ON (2x speedup)' if mixed_precision else 'OFF (FP32)'}")
    print(f"  VRAM Limit  : {f'{memory_limit_mb}MB' if memory_limit_mb else 'Dynamic (grow as needed)'}")
    print(f"  TF32        : ON (Ampere tensor cores)")
    print(f"  XLA JIT     : ON (op fusion)")
    print(f"  TF version  : {tf.__version__}")
    print(f"{'='*55}\n")


def get_optimal_batch_size(model_type: str = "transformer") -> int:
    """
    Returns optimal batch size for RTX 3050 4GB VRAM.
    Larger batches = faster training but more VRAM.

    RTX 3050 4GB VRAM budget:
      - Model weights  : ~50-100 MB
      - Gradients      : ~same as weights
      - Activations    : batch_size dependent
      - Optimizer state: ~2x weights

    Safe estimates for our model with LOOKBACK=60, features=46:
    """
    sizes = {
        "transformer": 256,   # Our full model — safe for 4GB
        "lstm_only":   512,
        "cnn_only":    1024,
        "small":       512,
    }
    return sizes.get(model_type, 256)


def check_cuda_installation():
    """Verify CUDA is properly installed for RTX 3050."""
    import tensorflow as tf
    import subprocess

    print(f"\n{'='*55}")
    print(f"  CUDA INSTALLATION CHECK")
    print(f"{'='*55}")
    print(f"  TF Built with CUDA : {tf.test.is_built_with_cuda()}")
    print(f"  GPU Available      : {tf.test.is_gpu_available()}")  # noqa

    gpus = tf.config.list_physical_devices("GPU")
    print(f"  Physical GPUs      : {len(gpus)}")
    for g in gpus:
        print(f"    -> {g}")

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,cuda_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"\n  nvidia-smi output:")
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
        else:
            print("  nvidia-smi not found or failed")
    except Exception:
        print("  nvidia-smi not accessible")

    print(f"{'='*55}\n")

    if not gpus:
        print(
            "\nGPU NOT FOUND. Install steps for RTX 3050:\n"
            "  1. Install CUDA 11.8:\n"
            "     https://developer.nvidia.com/cuda-11-8-0-download-archive\n"
            "  2. Install cuDNN 8.6 for CUDA 11.x:\n"
            "     https://developer.nvidia.com/rdp/cudnn-archive\n"
            "  3. Install TF with GPU support:\n"
            "     pip install tensorflow[and-cuda]==2.15.0\n"
            "  4. Verify: python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"\n"
        )


def benchmark_gpu(lookback: int = 60, n_features: int = 46, n_samples: int = 2048):
    """
    Quick benchmark to verify GPU is working and estimate training speed.
    Should see >500 samples/sec on RTX 3050.
    """
    import time
    import numpy as np
    import tensorflow as tf

    if not setup_gpu(verbose=False):
        print("GPU not available for benchmark")
        return

    print(f"\nBenchmarking RTX 3050 (batch forward pass)...")
    X = np.random.randn(n_samples, lookback, n_features).astype(np.float32)

    # Simple model for benchmarking
    inp = tf.keras.Input(shape=(lookback, n_features))
    x   = tf.keras.layers.LSTM(128, return_sequences=False)(inp)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="categorical_crossentropy")

    # Warmup
    _ = m.predict(X[:64], verbose=0)

    # Benchmark
    t0    = time.perf_counter()
    _     = m.predict(X, batch_size=256, verbose=0)
    elapsed = time.perf_counter() - t0
    throughput = n_samples / elapsed

    print(f"  Samples    : {n_samples}")
    print(f"  Time       : {elapsed:.2f}s")
    print(f"  Throughput : {throughput:.0f} samples/sec")
    print(f"  Status     : {'GOOD (GPU working)' if throughput > 300 else 'SLOW (check CUDA)'}")


if __name__ == "__main__":
    check_cuda_installation()
    benchmark_gpu()
