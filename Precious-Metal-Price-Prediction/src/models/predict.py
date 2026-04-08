import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pickle


# FILE PATHS
GOLD_MODEL_FILE = "models/model.pkl"
SILVER_MODEL_FILE = "models/silver_model.pkl"
USD_MODEL_FILE = "models/usd_model.pkl"

DATA_FILE = "data/processed/final_data.csv"



# LOAD MODELS
def load_models():
    with open(GOLD_MODEL_FILE, "rb") as f:
        gold_model = pickle.load(f)

    with open(SILVER_MODEL_FILE, "rb") as f:
        silver_model = pickle.load(f)

    with open(USD_MODEL_FILE, "rb") as f:
        usd_model = pickle.load(f)

    return gold_model, silver_model, usd_model

# LOAD DATA
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# GOLD FEATURES

def prepare_gold_features(df):
    df['Lag_1'] = df['Gold_24K_1g'].shift(1)
    df['Lag_2'] = df['Gold_24K_1g'].shift(2)
    df['Lag_3'] = df['Gold_24K_1g'].shift(3)

    df['MA_7'] = df['Gold_24K_1g'].rolling(7).mean()
    df['MA_30'] = df['Gold_24K_1g'].rolling(30).mean()

    df['USD_Change'] = df['USD_INR'].pct_change()
    df['Silver_Change'] = df['Silver_1g'].pct_change()

    df['DayOfWeek'] = df['Date'].dt.dayofweek

    df = df.dropna()
    return df

# SILVER FEATURES

def prepare_silver_input(df):
    lag1 = df['Silver_1g'].iloc[-1]
    lag2 = df['Silver_1g'].iloc[-2]
    lag3 = df['Silver_1g'].iloc[-3]

    ma7 = df['Silver_1g'].tail(7).mean()
    ma30 = df['Silver_1g'].tail(30).mean()

    return_val = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    momentum = lag1 - lag2

    ema_10 = df['Silver_1g'].ewm(span=10).mean().iloc[-1]
    rolling_std = df['Silver_1g'].rolling(5).std().iloc[-1]

    X = pd.DataFrame([{
        "Lag_1": lag1,
        "Lag_2": lag2,
        "Lag_3": lag3,
        "MA_7": ma7,
        "MA_30": ma30,
        "Return": return_val,
        "Momentum": momentum,
        "Gold_Influence": df['Gold_24K_1g'].iloc[-1],
        "EMA_10": ema_10,
        "Rolling_STD": rolling_std
    }])

    return X

# USD FEATURES
def prepare_usd_input(df):
    lag1 = df['USD_INR'].iloc[-1]
    lag2 = df['USD_INR'].iloc[-2]
    lag3 = df['USD_INR'].iloc[-3]

    ma3 = df['USD_INR'].tail(3).mean()
    ma7 = df['USD_INR'].tail(7).mean()

    X = pd.DataFrame([[lag1, lag2, lag3, ma3, ma7]],
                    columns=['Lag_1','Lag_2','Lag_3','MA_3','MA_7'])

    return X



# NEXT DAY PREDICTION

def predict_all():

    gold_model, silver_model, usd_model = load_models()
    df = load_data()

    # GOLD
    df_gold = prepare_gold_features(df.copy())
    last = df_gold.iloc[-1]

    X_gold = pd.DataFrame([[ 
        last['Lag_1'], last['Lag_2'], last['Lag_3'],
        last['MA_7'], last['MA_30'],
        last['USD_INR'], last['USD_Change'],
        last['Silver_1g'], last['Silver_Change'],
        last['Gold_22K_1g'],
        last['DayOfWeek']
    ]], columns=[
        'Lag_1','Lag_2','Lag_3',
        'MA_7','MA_30',
        'USD_INR','USD_Change',
        'Silver_1g','Silver_Change',
        'Gold_22K_1g','DayOfWeek'
    ])

    gold_pred = gold_model.predict(X_gold)[0]

    # SILVER
    X_silver = prepare_silver_input(df)
    silver_pred = silver_model.predict(X_silver)[0]

    # USD
    X_usd = prepare_usd_input(df)
    usd_pred = usd_model.predict(X_usd)[0]

    return gold_pred, silver_pred, usd_pred

def predict_next():
    gold, _, _ = predict_all()
    return gold


def predict_future(days=7):

    gold_model, silver_model, usd_model = load_models()
    df = load_data()

    preds = []

    temp_df = df.copy()

    for _ in range(days):

        # Prepare features again (IMPORTANT)
        temp_df = prepare_gold_features(temp_df)

        last = temp_df.iloc[-1]

        X = pd.DataFrame([[ 
            last['Lag_1'], last['Lag_2'], last['Lag_3'],
            last['MA_7'], last['MA_30'],
            last['USD_INR'], last['USD_Change'],
            last['Silver_1g'], last['Silver_Change'],
            last['Gold_22K_1g'],
            last['DayOfWeek']
        ]], columns=[
            'Lag_1','Lag_2','Lag_3',
            'MA_7','MA_30',
            'USD_INR','USD_Change',
            'Silver_1g','Silver_Change',
            'Gold_22K_1g','DayOfWeek'
        ])

        next_pred = gold_model.predict(X)[0]
        next_pred = next_pred * np.random.uniform(0.999, 1.001)

        preds.append(round(next_pred, 2))

        # 🔥 IMPORTANT: append prediction back
        new_row = temp_df.iloc[-1].copy()
        new_row['Gold_24K_1g'] = next_pred
        new_row['Date'] = new_row['Date'] + pd.Timedelta(days=1)

        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

    return preds
# MAIN
if __name__ == "__main__":

    print("Running Predictions...\n")

    gold, silver, usd = predict_all()

    print(f"🪙 Gold Prediction per 1g: ₹ {gold:.2f}")
    print(f"🔘 Silver Prediction per 1g: ₹ {silver:.2f}")
    print(f"💱 USD Prediction per INR: ₹ {usd:.2f}")

    print("\nDone")
