# Predicting Barclays Next-Day Stock Price Direction

## Project Overview

This project investigates whether historical price-based and technical features can be used to predict the **next trading day's closing-price direction for Barclays PLC (BARC.L)**.

Daily market data from **2015–2024** is used to construct a binary classification problem:

- **1** — the next trading day's closing price is higher than the current day's close.
- **0** — the next trading day's closing price is lower than or equal to the current day's close.

The project covers exploratory data analysis, feature engineering, model development, time-series cross-validation, hyperparameter tuning and out-of-sample evaluation.

## Project Structure

The analysis is divided into three notebooks:

### `01_exploratory_data_analysis.ipynb`

Explores the Barclays daily market data before modelling, including:

- data quality and sample coverage;
- closing-price behaviour;
- daily log-return distributions;
- rolling volatility;
- autocorrelation in returns and squared returns; and
- the distribution and stability of the prediction target.

The exploratory analysis identifies heavy-tailed returns, volatility clustering and little short-term linear dependence in raw returns.

### `02_feature_engineering.ipynb`

Constructs the modelling dataset using information available up to each trading day.

The final dataset contains **11 predictors** representing:

- relative price trends;
- current and lagged returns;
- rolling volatility; and
- price momentum using the Relative Strength Index (RSI).

Relative moving-average features are used instead of absolute moving-average levels to reduce dependence on the Barclays share-price level.

The notebook also checks the construction of the next-day target to prevent information from the future leaking into the predictors.

### `03_modelling_and_evaluation.ipynb`

Compares three classification models:

- Logistic Regression
- Random Forest
- XGBoost

The models are compared with simple majority-class and directional-persistence baselines.

Because this is a forecasting problem, observations are kept in chronological order. Model selection and hyperparameter tuning use **expanding-window time-series cross-validation**, followed by evaluation on a later **2023–2024 holdout period**.

Performance is assessed using several classification metrics, including:

- balanced accuracy;
- F1 score;
- ROC-AUC; and
- Matthews correlation coefficient (MCC).

Confusion matrices, ROC curves, performance through time and post-hoc permutation importance are also used to examine model behaviour.

## Key Results

The models show limited ability to predict next-day Barclays price direction.

During tuned time-series cross-validation, balanced accuracy is approximately **0.515–0.530** across the three models.

Performance on the later 2023–2024 holdout period is weaker. Random Forest achieves the highest holdout balanced accuracy at approximately **0.502**, with an ROC-AUC of **0.501** and MCC of **0.004**, all effectively at chance level.

XGBoost provides an example of why chronological out-of-sample evaluation is important: despite achieving the strongest tuned cross-validation balanced accuracy (**0.530**), its holdout balanced accuracy falls to approximately **0.474**.

Overall, the results provide **no convincing evidence that the engineered technical features reliably predict next-day Barclays closing-price direction out of sample**.

## Installation

Clone the repository and install the required Python packages.

```bash
git clone https://github.com/keeratsingh743/stock_direction_prediction.git
cd stock_direction_prediction
pip install -r requirements.txt
```

The main Python packages used in the analysis include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
yfinance
```

## Running the Project

Run the notebooks in numerical order:

```text
01_exploratory_data_analysis.ipynb
02_feature_engineering.ipynb
03_modelling_and_evaluation.ipynb
```

The first notebook obtains and explores the market data, the second constructs the modelling features and dataset, and the third performs model training and evaluation.

## Disclaimer

This project is intended for statistical and educational purposes only. It does not constitute financial or investment advice.
