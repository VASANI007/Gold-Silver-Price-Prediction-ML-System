import pandas as pd
import pickle
import numpy as np
import os

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def train_silver_model():

    df = pd.read_csv("data/processed/final_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    #  CLEAN FEATURES (IMPORTANT)

    df['Lag_1'] = df['Silver_1g'].shift(1)
    df['Lag_2'] = df['Silver_1g'].shift(2)
    df['Lag_3'] = df['Silver_1g'].shift(3)
    
    df['MA_7'] = df['Silver_1g'].rolling(7).mean()
    df['MA_30'] = df['Silver_1g'].rolling(30).mean()

    # Light features (safe)
    df['Return'] = df['Silver_1g'].pct_change()
    df['Momentum'] = df['Silver_1g'] - df['Silver_1g'].shift(2)
    
    # Gold influence (keep simple)
    df['Gold_Influence'] = df['Gold_24K_1g']

    df['Target'] = df['Silver_1g'].shift(-1)
    
    df['EMA_10'] = df['Silver_1g'].ewm(span=10).mean()
    df['Rolling_STD'] = df['Silver_1g'].rolling(5).std()
    df = df.dropna()


    # FEATURES

    X = df[
        [
            'Lag_1','Lag_2','Lag_3',
            'MA_7','MA_30',
            'Return','Momentum',
            'Gold_Influence',
            'EMA_10','Rolling_STD'
        ]
    ]

    y = df['Target']

    #  MODEL (BEST FOR THIS CASE)
    model = Ridge(alpha=0.05)

    # CROSS VALIDATION
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


    # METRICS

    final_mae = float(np.mean(mae_list))
    final_r2 = float(np.mean(r2_list))
    final_rmse = float(np.mean(rmse_list))

    print("\n Stable Silver Model Performance:")
    print(f"MAE: {final_mae:.2f}")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"RMSE: {final_rmse:.2f}")

    # SAVE
    metrics = {
        "MAE": final_mae,
        "RMSE": final_rmse,
        "R2": final_r2
    }

    os.makedirs("models", exist_ok=True)

    with open("models/silver_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    model.fit(X, y)

    with open("models/silver_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model fixed & saved")
    return model


if __name__ == "__main__":
    print("Training Fixed Silver Model...\n")
    train_silver_model()
    print("\nAll Done!")