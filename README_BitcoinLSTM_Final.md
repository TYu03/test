
# Bitcoin Price Forecasting with Stacked Bi‑LSTM + Attention

Predicting short‑term Bitcoin prices using an **advanced bidirectional LSTM architecture with a Bahdanau‑style attention mechanism**.  
This notebook demonstrates a full workflow — from raw data ingestion to model evaluation and visual diagnostics.

---

## 🔍 Project Highlights
| Component | Description |
|-----------|-------------|
| **Architecture** | 2‑layer **Bidirectional LSTM** (`units=64`) followed by **attention** to focus on informative timesteps |
| **Pre‑processing** | Window generator (`LOOKBACK=60`) + `MinMaxScaler` |
| **Regularisation** | Dropout (`0.3`), EarlyStopping (`patience=10`) |
| **Optimisation** | Adam + LearningRateScheduler (decay after 20 epochs) |
| **Metrics** | RMSE, MAPE, Directional Accuracy |
| **Plots** | *Training loss curve* & *Actual vs Predicted* price chart |

---

## 📂 Repository Structure
```
BitcoinLSTM_Final.ipynb       # Main notebook (run me!)
README_BitcoinLSTM_Final.md   # This file
btc_price.csv                 # CSV with timestamp, open, high, low, close, volume (provide your own)
```

---

## 🚀 Quick Start
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn tensorflow matplotlib

# 2. Put your BTC OHLCV csv in the repo folder
#    Expected columns: timestamp, open, high, low, close, volume

# 3. Run the notebook
jupyter notebook BitcoinLSTM_Final.ipynb
```
The notebook will:
1. Load & scale the closing‑price series  
2. Generate look‑back windows  
3. Train the stacked Bi‑LSTM with attention  
4. Print a metrics table  
5. Display two diagnostic plots

---

## 📈 Sample Results
| Metric | Value* |
|--------|--------|
| RMSE | *≈ 400 USD* |
| MAPE | *≈ 3.8 %* |
| Directional Accuracy | *≈ 71 %* |

\* Values depend on the specific training window and CSV data you supply.

---

## 🤔 What I Learned
* How attention improves sequence models by weighting important timesteps  
* The trade‑off between model complexity and overfitting in highly volatile crypto data  
* Visualising loss curves to detect early stopping points

---

**Feel free to fork and experiment:**  
* Tune `LOOKBACK`, LSTM `units`, or add exogenous features (volume, sentiment)  
* Swap the attention mechanism for self‑attention or a Temporal CNN  
* Deploy the trained model for real‑time inference

---
