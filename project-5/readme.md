# Momentum Trading on the S&P 500

This project is an introduction to building a **momentum-based trading strategy** for the **S&P 500 (Standard & Poor's 500)** — a stock market index that tracks the performance of 500 of the largest publicly traded companies in the United States.  
The S&P 500 is widely regarded as one of the best representations of the U.S. stock market and economy. Over the long term, it has shown consistent growth, but can also experience significant short-term volatility.

In this project, we make a first attempt to design, test, and evaluate a **momentum strategy** that reacts to recent price trends.  
By the end, you will have a working framework that can be expanded and customised to suit your future trading strategy development needs.

We use  NumPy, SciPy, SQLite and related Python packages for data handling, simulation, and analysis.

---

## Core Components

- ⚡ **Momentum Strategy Logic**
  - Uses recent price changes to determine trade direction (trend-following)
  - Stores portfolio positions and cash in a local SQLite database

- 📈 **Geometric Brownian Motion (GBM) Model**
  - Custom `GBM` class to simulate and forecast price paths
  - Methods:
    - `calibrate(trajectory, Dt)` — estimate drift & volatility from data
    - `simulate(N, K, Dt, S0)` — simulate price trajectories
    - `forecast(latest, T, confidence)` — predict price intervals
    - `expected_shortfall(T, confidence)` — placeholder for risk metrics

- 💾 **Data Storage**
  - `prepare()` function imports historical S&P 500 prices from `SP500.csv` into local SQLite tables:
    - `prices(theday, price)`
    - `positions(time_of_trade, instrument, quantity, cash)`

---

## Environment Setup

1. **Create and activate the Conda environment**
```bash
conda create --name und-ai-trading-strategies-p5 python=3.10
conda activate und-ai-trading-strategies-p5
```

2. Install dependencies
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
Project_Learner.ipynb
```

3. **Load price data into the local database**
```python
prepare()
```

4. **Build and evaluate your first momentum strategy**

* Generate features
* Train and test models
* Simulate trades and evaluate portfolio value

5. **Experiment with GBM forcasts**
```python
model = GBM()
model.mu = 0.25
model.sigma = 0.1
print(model.forecast(100, 0.5, 0.9))
```

## Project Sturcture

```bash
project/
│
└── readme.md               # Project documentation
├── Project_Learner.ipynb   # Main notebook for feature engineering, model training & evaluation
├── SP500.csv               # SP500 historical data
├── requirements.txt        # Python dependencies
```
