import pandas as pd
import numpy as np
import pickle
import os

# FILE PATHS
MODEL_FILE = "models/model.pkl"
DATA_FILE = "data/processed/final_data.csv"


# LOAD MODEL
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model not found. Run train_model.py first")

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    return model


# LOAD DATA
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# PREPARE FEATURES (SAME AS TRAIN)
def prepare_features(df):

    df = df[['Date', 'Gold_24K_1g', 'Gold_22K_1g', 'Silver_1g', 'USD_INR']].dropna()

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


# NEXT DAY PREDICTION
def predict_next():

    model = load_model()
    df = load_data()
    df = prepare_features(df)

    last = df.iloc[-1]

    X_new = pd.DataFrame([[
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

    prediction = model.predict(X_new)[0]

    return prediction


# FUTURE PREDICTION
def predict_future(days=7):

    model = load_model()
    df = load_data()

    predictions = []

    for _ in range(days):

        df = prepare_features(df)
        last = df.iloc[-1]

        X_new = pd.DataFrame([[
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

        pred = model.predict(X_new)[0]
        predictions.append(pred)

        #  Append prediction for next iteration
        new_row = {
            'Date': last['Date'] + pd.Timedelta(days=1),
            'Gold_24K_1g': pred,
            'Gold_22K_1g': pred * (22/24),
            'Silver_1g': last['Silver_1g'],
            'USD_INR': last['USD_INR']
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return predictions


# MAIN
if __name__ == "__main__":

    print("Running Predictions...\n")

    next_price = predict_next()
    print(f"Next Day Prediction: ₹ {next_price:.2f}")

    future_prices = predict_future(7)

    print("\nNext 7 Days Forecast:")
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: ₹ {price:.2f}")

    print("\nDone")