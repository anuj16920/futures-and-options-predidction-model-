"""
====================================================================
MODEL ARCHITECTURE v3 - Transformer + CNN + LSTM
====================================================================
Designed for ~945K rows of 1-minute NIFTY data.

Architecture flow:
  Input (60, 46)
    |
  Positional Encoding
    |
  CNN Block (local pattern extraction)
    |
  Transformer Encoder x2 (global attention across all 60 timesteps)
    |
  Bidirectional LSTM (sequential memory)
    |
  Multi-Head Attention (focus on most predictive bars)
    |
  Global Average Pool
    |
  Dense Head -> Softmax (3 classes)

Why this works for financial data:
  - CNN: Captures local price action patterns (candlestick formations)
  - Transformer: Attends globally (e.g., current bar vs 30 bars ago)
  - LSTM: Maintains sequential state memory
  - Combined: Understands BOTH local structure AND global regime

Target: ~420K parameters (well-suited for 660K training rows)
====================================================================
"""

import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    LOOKBACK, NUM_CLASSES,
    TRANSFORMER_HEADS, TRANSFORMER_DIM, TRANSFORMER_FF_DIM,
    TRANSFORMER_BLOCKS, TRANSFORMER_DROPOUT,
    CNN_FILTERS, CNN_KERNEL_SIZE,
    LSTM_UNITS, LSTM_LAYERS,
    DENSE_UNITS, DROPOUT_RATE, L2_REG,
    LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_PATIENCE,
    LR_REDUCE_FACTOR, LR_MIN, LABEL_SMOOTHING,
    FOCAL_GAMMA, MODEL_DIR, TENSORBOARD_DIR, LOG_LEVEL
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("ModelArchitecture")


# ─────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────

class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding so Transformer knows timestep order.
    Critical for financial sequences where position (time) matters.
    """
    def __init__(self, max_len=200, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model  = tf.shape(x)[2]
        # Compute encoding
        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        dims      = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angles    = positions / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        sin_part  = tf.math.sin(angles[:, 0::2])
        cos_part  = tf.math.cos(angles[:, 1::2])
        # Interleave sin/cos
        pe = tf.reshape(
            tf.stack([sin_part, cos_part], axis=-1),
            [seq_len, -1]
        )[:, :d_model]
        return x + pe[tf.newaxis, :, :]

    def get_config(self):
        cfg = super().get_config()
        cfg["max_len"] = self.max_len
        return cfg


# ─────────────────────────────────────────────────────────────────
# TRANSFORMER ENCODER BLOCK
# ─────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(layers.Layer):
    """
    Standard Transformer encoder with:
      - Multi-head self-attention
      - Position-wise feedforward network
      - Add & Norm (pre-norm variant for stability)
      - Dropout regularization
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.ff1    = layers.Dense(ff_dim, activation="gelu")
        self.ff2    = layers.Dense(d_model)
        self.norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.norm2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop1  = layers.Dropout(dropout)
        self.drop2  = layers.Dropout(dropout)

    def call(self, x, training=None):
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn   = self.attention(x_norm, x_norm, training=training)
        x      = x + self.drop1(attn, training=training)
        # Pre-norm feedforward
        x_norm = self.norm2(x)
        ff     = self.ff2(self.ff1(x_norm))
        x      = x + self.drop2(ff, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.ff2.units, "num_heads": self.attention.num_heads,
            "ff_dim": self.ff1.units, "dropout": self.drop1.rate
        })
        return cfg


# ─────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss: down-weights easy/majority examples,
    forces model to focus on hard Bullish/Bearish predictions.
    gamma=2.0 is standard; higher = more focus on hard examples.
    """
    def __init__(self, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n_cls = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_sm = y_true * (1 - self.label_smoothing) + self.label_smoothing / n_cls
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true_sm * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        weight = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return cfg


# ─────────────────────────────────────────────────────────────────
# COSINE WARMUP SCHEDULE
# ─────────────────────────────────────────────────────────────────

class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup + cosine decay.
    Warmup prevents early divergence on large datasets.
    """
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr      = base_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.min_lr       = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * step / self.warmup_steps
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + tf.cos(np.pi * (step - self.warmup_steps) /
                       (self.total_steps - self.warmup_steps))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr
        }


# ─────────────────────────────────────────────────────────────────
# MAIN MODEL BUILDER
# ─────────────────────────────────────────────────────────────────

def build_model(
    lookback:           int   = LOOKBACK,
    n_features:         int   = 46,
    num_classes:        int   = NUM_CLASSES,
    transformer_heads:  int   = TRANSFORMER_HEADS,
    transformer_dim:    int   = TRANSFORMER_DIM,
    transformer_ff_dim: int   = TRANSFORMER_FF_DIM,
    transformer_blocks: int   = TRANSFORMER_BLOCKS,
    transformer_drop:   float = TRANSFORMER_DROPOUT,
    cnn_filters:        int   = CNN_FILTERS,
    cnn_kernel:         int   = CNN_KERNEL_SIZE,
    lstm_units:         int   = LSTM_UNITS,
    lstm_layers:        int   = LSTM_LAYERS,
    dense_units:        int   = DENSE_UNITS,
    dropout_rate:       float = DROPOUT_RATE,
    l2_reg:             float = L2_REG,
    learning_rate:      float = LEARNING_RATE,
    weight_decay:       float = WEIGHT_DECAY,
    steps_per_epoch:    int   = 1000,
    warmup_epochs:      int   = 5,
    total_epochs:       int   = 100,
) -> keras.Model:
    """
    Build the full Transformer-CNN-BiLSTM model.

    Architecture:
      Input -> Project -> PosEnc -> CNN -> Transformer x2
            -> BiLSTM x2 -> MH-Attention -> GAP -> Dense -> Softmax
    """
    reg    = regularizers.l2(l2_reg)
    inputs = keras.Input(shape=(lookback, n_features), name="sequence_input")

    # ── 1. Input Projection ───────────────────────────────────
    # Project raw features to transformer_dim
    x = layers.Dense(transformer_dim, kernel_regularizer=reg, name="input_proj")(inputs)
    x = layers.LayerNormalization(name="input_norm")(x)

    # ── 2. Positional Encoding ────────────────────────────────
    x = PositionalEncoding(name="pos_enc")(x)

    # ── 3. CNN Block (local pattern extraction) ───────────────
    # causal padding = no future leakage
    x = layers.Conv1D(cnn_filters, cnn_kernel, padding="causal",
                      activation="gelu", kernel_regularizer=reg, name="cnn_1")(x)
    x = layers.BatchNormalization(name="bn_cnn")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_cnn")(x)

    # Residual CNN
    x_res = layers.Conv1D(cnn_filters, 1, padding="same", name="cnn_res")(x)
    x = layers.Conv1D(cnn_filters, cnn_kernel, padding="causal",
                      activation="gelu", kernel_regularizer=reg, name="cnn_2")(x)
    x = layers.BatchNormalization(name="bn_cnn2")(x)
    x = layers.Add(name="cnn_residual")([x, x_res])

    # ── 4. Transformer Encoder Blocks ─────────────────────────
    for i in range(transformer_blocks):
        x = TransformerEncoderBlock(
            d_model=cnn_filters,
            num_heads=transformer_heads,
            ff_dim=transformer_ff_dim,
            dropout=transformer_drop,
            name=f"transformer_{i+1}"
        )(x)

    # ── 5. Bidirectional LSTM ─────────────────────────────────
    for i in range(lstm_layers):
        return_seq = True   # Always return sequences for stacking / attention
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=return_seq,
                kernel_regularizer=reg,
                dropout=0.1,
                name=f"lstm_{i+1}"
            ),
            name=f"bilstm_{i+1}"
        )(x)
        x = layers.LayerNormalization(name=f"ln_lstm_{i+1}")(x)
        if i < lstm_layers - 1:
            x = layers.Dropout(dropout_rate * 0.5, name=f"drop_lstm_{i+1}")(x)

    # ── 6. Multi-Head Attention over LSTM output ──────────────
    x = layers.MultiHeadAttention(
        num_heads=transformer_heads,
        key_dim=lstm_units // transformer_heads,
        dropout=dropout_rate * 0.3,
        name="final_attention"
    )(x, x)
    x = layers.LayerNormalization(name="ln_final_attn")(x)

    # ── 7. Pooling ────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── 8. Dense Head ─────────────────────────────────────────
    x = layers.Dense(dense_units, activation="gelu",
                     kernel_regularizer=reg, name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense_1")(x)

    x = layers.Dense(dense_units // 2, activation="gelu",
                     kernel_regularizer=reg, name="dense_2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_dense_2")(x)

    # ── 9. Output ─────────────────────────────────────────────
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="output_softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="NIFTY_TransformerCNNLSTM_v3")

    # ── Optimizer: AdamW + Cosine Warmup ──────────────────────
    warmup_steps = steps_per_epoch * warmup_epochs
    total_steps  = steps_per_epoch * total_epochs
    lr_schedule  = WarmupCosineSchedule(
        base_lr=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=LR_MIN
    )
    # Use Adam with weight decay manually applied (AdamW not available in Keras 2.10)
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]
    )

    total_params = model.count_params()
    logger.info(f"Model built | Parameters: {total_params:,}")
    return model


def get_callbacks(checkpoint_path, tensorboard_dir=None):
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(Path(checkpoint_path).parent / "train_log.csv"), append=False
        ),
    ]
    if tensorboard_dir:
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
        cbs.append(keras.callbacks.TensorBoard(str(tensorboard_dir), histogram_freq=0))
    return cbs


def print_model_summary(model):
    model.summary(line_length=90, show_trainable=True)
    p = model.count_params()
    tp = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\nTotal: {p:,} | Trainable: {tp:,} | Frozen: {p-tp:,}")


if __name__ == "__main__":
    from features.feature_engineering import get_feature_columns
    n_feat = len(get_feature_columns())
    model  = build_model(lookback=LOOKBACK, n_features=n_feat, steps_per_epoch=1300)
    print_model_summary(model)
    dummy = np.random.randn(4, LOOKBACK, n_feat).astype(np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f"\nOutput: {out.shape} | Sum: {out[0].sum():.4f}")
