# Stock Trading with Machine Learning

This project explores using **machine learning models** to predict stock returns and evaluate trading strategies. It extends beyond the original Deep Q-Network (DQN) approach by incorporating classical ML models, feature engineering, and performance evaluation.

## Features & Methods

- 📈 **Supervised Learning Models** — Random Forest, Gradient Boosting, Logistic Regression.
- 📊 **Feature Engineering** —
- Technical indicators: Bollinger Bands (bb_bbm, bb_bbw, bb_bbli, etc.), RSI, IBS.
- Return-based features: 1-day, 5-day, 10-day, 20-day historical returns.
- Cyclical features: month (sin/cos encoding), weekdays (one-hot encoded).
- Macro/sentiment features: VIX, Google Trends (recession), volume (log transformed).
- 🧹 **Data Preprocessing** — Handling missing values, dropping low-importance features, baseline comparisons.
- 🔍 **Model Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix, feature importance.
- 📓 **Interactive Exploration** — Implemented in a Jupyter Notebook for transparency and experimentation.

## Environment Setup

1. **Create and activate the Conda environment:**

```bash
conda create --name und-ai-trading-strategies-p4 python=3.10
conda activate und-ai-trading-strategies-p4
```

2. **Install required packeages:**
```bash
pip install -r requirements.txt
```

## Running the Project
1. **Launch Jupyter Lab**

```bash
jupyter lab
```

2. **Open Notebook**
```bash
project_starter.ipynb
```

3. **Explore Features & Train Models**
* Generate engineered features from stock price and macro data.
* Train Random Forests and other classifiers.
* Tune hyperparameters and adjust decision thresholds.


4. **Evaluate Models**
* Compare against baseline accuracy (majority class predictor).
* Assess precision/recall trade-offs.
* Visualize learning curves and feature importance.

## Project Sturcture

```bash
project/
│
├── GoogleTrendsData.csv    # Google Trends macro data
├── vix_data.csv            # VIX volatility index data
├── xlv_data.csv            # XLV sector ETF data
├── project_starter.ipynb   # Main notebook for feature engineering, model training & evaluation
├── requirements.txt        # Python dependencies
└── readme.md               # Project documentation
```
