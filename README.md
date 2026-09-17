# ✈️ Turbofan Engine Remaining Useful Life (RUL) Prediction

Predicting how many operating cycles a jet engine has left before failure — using NASA's CMAPSS turbofan degradation simulation dataset.

## 📌 Overview

Predictive maintenance is one of the highest-value applications of machine learning in industry — knowing *when* a machine will fail lets you schedule maintenance before it actually breaks, instead of reacting after the fact.

This project builds regression models that predict the **Remaining Useful Life (RUL)** — the number of cycles left before failure — for a fleet of simulated turbofan engines, using raw sensor telemetry (temperatures, pressures, fan speeds, etc.) collected over each engine's lifetime.

The dataset comes in **four subsets (FD001–FD004)**, each progressively harder: more operating conditions, more fault modes, more noise. Solving all four means the pipeline has to handle real operating-condition drift, not just a single clean scenario.

---

## 🗂️ Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** — a widely used benchmark for RUL prediction research.

| Subset | Operating Conditions | Fault Modes | Engines (train) |
|:------:|:---------------------:|:-----------:|:----------------:|
| FD001  | 1                     | 1           | 100              |
| FD002  | 6                     | 1           | 260              |
| FD003  | 1                     | 2           | 100              |
| FD004  | 6                     | 2           | 249              |

Each row is one engine's sensor reading at one operating cycle. Engines run until failure in the training data; in the test data they're cut off partway through, and the goal is to predict how many cycles are left.

---

## 🔧 Approach

The pipeline was built up incrementally, subset by subset, with each step validated before moving to the next:

1. **Exploratory analysis** — identified which of the 21 sensors actually carry signal (several are constant or near-constant and were dropped).
2. **RUL labeling** — computed ground-truth RUL for training data as `max_cycle − current_cycle` per engine.
3. **RUL capping** — capped RUL at a per-dataset threshold. Early in an engine's life, degradation hasn't started yet, so trying to predict "180 cycles left" vs. "179 cycles left" adds noise without adding real signal. Capping focuses the model on the region where sensors actually start to show wear.
4. **Operating-condition normalization** (FD002 & FD004 only) — these subsets cycle through **6 distinct operating regimes**, and raw sensor values shift with the regime, not just with wear. Sensor values were clustered by operating setting (K-Means) and z-normalized *within each condition*, so the model sees wear trends instead of regime noise.
5. **Rolling-window features** — rolling mean/std over each sensor per engine, to capture *trends* in degradation rather than single noisy readings.
6. **Sensor degradation analysis** — computed each sensor's correlation with cycle count, per engine, to quantify which sensors degrade consistently vs. which are just noise (see chart below).
7. **Model comparison** — Random Forest, XGBoost, and LightGBM were benchmarked on each subset; the best performer per subset was carried forward to final testing.
8. **Evaluation on held-out test engines** — using the official NASA test/RUL files, predicting from only the *last available cycle* per test engine (the realistic "engine is still running, what's left?" scenario).

---

## 📊 Sensor Degradation Signal (FD001)

Correlation of each sensor's raw value with cycle count, averaged across all 100 training engines — a quick way to separate genuinely informative sensors from noise before feature engineering:

| Sensor | \|mean correlation\| |
|:------:|:--------------------:|
| sensor_11 | 0.81 |
| sensor_12 | 0.79 |
| sensor_4  | 0.78 |
| sensor_7  | 0.76 |
| sensor_15 | 0.72 |
| sensor_6  | 0.10 *(near-zero — noise)* |

Sensors like 11, 12, 4, and 7 degrade steadily and predictably across almost every engine, while sensor_6 carries essentially no degradation signal — confirming the feature-selection choices made during exploration.

---

## 🏆 Results

Final test-set performance, per subset, using each subset's best model:

| Subset | Model | Test MAE (cycles) | Test RMSE (cycles) |
|:------:|:-----:|:------------------:|:-------------------:|
| FD001  | Random Forest | **12.03** | **17.15** |
| FD002  | LightGBM      | **14.72** | **19.62** |
| FD003  | LightGBM      | **15.99** | **21.74** |
| FD004  | XGBoost       | **19.09** | **25.10** |

**Reading the results:** error increases as the problem gets harder — FD001 (1 condition, 1 fault mode) is the easiest and most accurate, while FD004 (6 conditions, 2 fault modes) is the hardest and has the highest error. This is the expected pattern for CMAPSS and is itself a sanity check that the modeling was done correctly — a model that performed *equally well* across all four subsets would be more suspicious than one that degrades gracefully with task difficulty.

An average error of ~12–19 cycles, on engines that run for hundreds of cycles, is a solid result and in line with published benchmarks on this dataset.

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **Modeling** — scikit-learn (Random Forest), XGBoost, LightGBM
- **Clustering** — K-Means (operating-condition detection for FD002/FD004)
- **Visualization** — Matplotlib

---

## 📁 Repository Structure

```
├── data/
│   └── sensor_degradation_summary_FD00X.csv   # Sensor degradation summaries per subset
├── notebooks/
│   ├── 01_data_explor.ipynb   # FD001 — baseline pipeline & feature engineering
│   ├── 02_data_explor.ipynb   # FD002 — adds operating-condition normalization
│   ├── 03_data_explor.ipynb   # FD003 — single condition, dual fault mode
│   └── 04_data_explor.ipynb   # FD004 — hardest subset, 6 conditions + dual fault mode
└── README.md
```

> **Note:** The raw CMAPSS train/test/RUL `.txt` files are not stored in this repo (see `.gitignore`). Download them from the **[NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)** and place them in a local `data/raw/` folder before running the notebooks.

---

## 🚀 Key Takeaways

- Operating-condition normalization was the single biggest lever for FD002/FD004 — without it, sensor readings were dominated by which regime the engine was in, not how worn it was.
- RUL capping consistently reduced error more than any individual model swap — a good reminder that framing the target correctly often matters more than the algorithm.
- Gradient-boosted trees (XGBoost/LightGBM) edged out Random Forest on the harder, multi-condition subsets, but Random Forest was competitive and simpler on the single-condition ones.

---

## 🔭 Future Work

- Sequence models (LSTM/GRU) to capture degradation trends more directly than rolling-window statistics
- Hyperparameter tuning via Optuna/GridSearch for each subset's final model
- Unified single model trained across all four subsets with condition/fault-mode as input features
