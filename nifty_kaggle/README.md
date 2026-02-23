# NIFTY 50 Kaggle Predictor - Transformer-CNN-BiLSTM Model

Deep learning model for predicting NIFTY 50 index movements using 1-minute candle data with multi-timeframe feature engineering.

## 🎯 Model Architecture

**Hybrid Architecture:** Transformer + CNN + BiLSTM
- **Parameters:** 866,435
- **Input:** 60-minute lookback window with 46 features
- **Output:** 3-class prediction (Bullish, Bearish, Neutral)

### Architecture Flow:
```
Input (60, 46) 
  → Positional Encoding
  → CNN Block (local pattern extraction)
  → Transformer Encoder x2 (global attention)
  → Bidirectional LSTM x2 (sequential memory)
  → Multi-Head Attention
  → Dense Head → Softmax (3 classes)
```

## 📊 Dataset

**Source:** [NIFTY 50 Minute Data on Kaggle](https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data)

**Details:**
- ~1,018,000 rows of 1-minute OHLCV data
- Date range: 2015-01-09 to 2026-01-22 (11 years)
- 2,718 trading days
- Market hours: 09:15 - 15:29 IST

**Note:** Dataset not included in repository. Download from Kaggle and place `nifty_50_minute.csv` in `data/` folder.

## 🚀 Features

### Multi-Timeframe Analysis (1m, 5m, 15m)
- RSI, EMA (9/21/50), ATR, VWAP
- Bollinger Bands, MACD
- Volume features, Momentum
- Time encoding (sin/cos)
- Market session flags
- Intraday context features

### Label Generation
- ATR-adjusted dynamic thresholds
- No lookahead bias
- Balanced classes: ~40% Bullish, ~39% Bearish, ~21% Neutral

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nifty_kaggle.git
cd nifty_kaggle

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place nifty_50_minute.csv in data/ folder
```

## 🎮 Usage

### Training
```bash
# Check GPU
python main.py --mode gpu_check

# Train model
python main.py --mode train

# Train with force reload
python main.py --mode train --force_reload
```

### Evaluation
```bash
# Evaluate model
python main.py --mode evaluate

# Run backtest
python main.py --mode backtest

# Walk-forward validation
python main.py --mode wfv

# Run all
python main.py --mode all
```

### Fine-tuning
```bash
# Fine-tune on recent data
python main.py --mode finetune --strategy partial --recent_rows 50000
```

## ⚙️ Configuration

Edit `config/config.py` to customize:
- Model architecture parameters
- Training hyperparameters
- Feature engineering settings
- Data split ratios

## 🖥️ GPU Support

**Optimized for RTX 3050 (4GB VRAM)**
- Mixed precision FP16 training
- Memory-efficient sequence building
- Batch size optimization

**Requirements:**
- CUDA 11.0/11.2
- cuDNN 8.x
- TensorFlow 2.10+

## 📈 Training Details

**Optimizer:** Adam with Cosine Warmup Schedule
- Learning rate: 1e-3 → 1e-6
- Warmup: 5 epochs
- Weight decay: 1e-4

**Loss:** Focal Loss (γ=2.0) with label smoothing (0.05)

**Regularization:**
- Dropout: 0.3
- L2 regularization: 1e-4
- Early stopping: patience 15

**Data Split:**
- Train: 70% (2015-2022)
- Validation: 15% (2022-2024)
- Test: 15% (2024-2026)

## 📊 Expected Performance

- **Accuracy:** ~34-40%
- **Directional Accuracy (filtered):** ~50-55%
- **AUC:** ~0.51-0.52

## 🗂️ Project Structure

```
nifty_kaggle/
├── config/              # Configuration files
├── data/                # Data loading and processing
├── features/            # Feature engineering
├── models/              # Model architecture
├── training/            # Training pipeline
├── evaluation/          # Evaluation metrics
├── finetuning/          # Fine-tuning utilities
├── utils/               # GPU config and utilities
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
```

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA installation
nvcc --version

# Verify TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Issues
- Reduce `BATCH_SIZE` in config
- Use CPU if GPU memory insufficient
- Enable memory growth in `utils/gpu_config.py`

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- Dataset: [debashis74017 on Kaggle](https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data)
- Inspired by modern deep learning architectures for time series prediction
