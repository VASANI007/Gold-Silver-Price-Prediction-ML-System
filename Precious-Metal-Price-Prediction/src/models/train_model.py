import pandas as pd
import numpy as np
import os
import pickle

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# FILE PATHS
DATA_FILE = "data/processed/final_data.csv"
MODEL_FILE = "models/model.pkl"


# LOAD DATA
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# FEATURE ENGINEERING
def prepare_features(df):

    df = df[['Date', 'Gold_24K_1g', 'Gold_22K_1g', 'Silver_1g', 'USD_INR']].dropna()

    # Lag features (Gold)
    df['Lag_1'] = df['Gold_24K_1g'].shift(1)
    df['Lag_2'] = df['Gold_24K_1g'].shift(2)
    df['Lag_3'] = df['Gold_24K_1g'].shift(3)

    # Moving averages
    df['MA_7'] = df['Gold_24K_1g'].rolling(7).mean()
    df['MA_30'] = df['Gold_24K_1g'].rolling(30).mean()

    # USD change
    df['USD_Change'] = df['USD_INR'].pct_change()

    # Silver change (important 🔥)
    df['Silver_Change'] = df['Silver_1g'].pct_change()

    # Seasonality
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    df = df.dropna()

    return df


# TRAIN MODEL
def train_model():

    df = load_data()
    df = prepare_features(df)

    # Target = next day Gold price
    df['Target'] = df['Gold_24K_1g'].shift(-1)
    df = df.dropna()

    X = df[
        [
            'Lag_1', 'Lag_2', 'Lag_3',
            'MA_7', 'MA_30',
            'USD_INR',
            'USD_Change',
            'Silver_1g',
            'Silver_Change',
            'Gold_22K_1g',
            'DayOfWeek'
        ]
    ]

    y = df['Target']

    model = Ridge(alpha=1.0)

    tscv = TimeSeriesSplit(n_splits=5)

    mae_list, r2_list, rmse_list = [], [], []

    for train_idx, test_idx in tscv.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_list.append(mean_absolute_error(y_test, y_pred))
        r2_list.append(r2_score(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    print("\n Final Model Performance:")
    print(f"MAE: {np.mean(mae_list):.2f}")
    print(f"R2 Score: {np.mean(r2_list):.4f}")
    print(f"RMSE: {np.mean(rmse_list):.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("\n  Model saved → models/model.pkl")

    return model


# RUN
if __name__ == "__main__":
    print("Training Model...\n")
    train_model()
    print("\nAll Done!")