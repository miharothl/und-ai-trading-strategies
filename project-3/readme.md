# Stock Trading with Deep Q-Network (DQN)

This project implements a **Deep Q-Network (DQN)** reinforcement learning agent to trade stocks based on three technical analysis features:

- **Close** — Closing price of the stock
- **BB_upper** — Upper Bollinger Band
- **BB_lower** — Lower Bollinger Band

The system is built in a **Jupyter Notebook** environment, allowing you to explore, train, and evaluate the trading agent interactively.

## Project Features

- 📈 **Reinforcement Learning** — Uses a DQN to learn trading strategies from historical price data.
- 📊 **Technical Indicators** — Bollinger Bands are calculated and used as input features.
- 🛠 **Training & Testing** — Includes both model training and evaluation/testing routines.
- 💾 **Model Saving** — Saves model checkpoints during training for recovery or reuse.
- 📓 **Interactive Exploration** — All workflows are in a Jupyter Notebook for transparency and experimentation.

## Environment Setup

1. **Create and activate the Conda environment:**

```bash
conda create --name und-ai-trading-strategies-p3 python=3.10
conda activate und-ai-trading-strategies-p3
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
project_nb.ipynb
```

3. **Train the model**

Run the training cells — the agent will:
 * Observe the stock price and Bollinger Band features.
 * Decide when to buy, sell, or hold.
 * Update its strategy using experience replay.

4. **Test the model**
After training, run the testing section to:
 * Evaluate the model on unseen data.
 * Plot buy/sell actions against the true stock prices.
 * Report total profit, number of winning trades, and losses.

## Project Sturcture

```bash
project/
│
├── project_nb.ipynb                                    # Main notebook for training/testing the agent
├── GOOG_2009-2010_6m_RAW_1d.csv                        # Raw data required to train and test the agent
├── requirements.txt                                    # Python dependencies
└── README.md                                           # Project documentation
```