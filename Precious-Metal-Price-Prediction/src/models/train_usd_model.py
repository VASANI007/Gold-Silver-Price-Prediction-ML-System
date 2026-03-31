import pandas as pd
import pickle
from sklearn.linear_model import Ridge

# Load data
df = pd.read_csv("data/processed/final_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Only USD data
df = df[['Date', 'USD_INR']].dropna()

# Features
df['Lag_1'] = df['USD_INR'].shift(1)
df['Lag_2'] = df['USD_INR'].shift(2)
df['Lag_3'] = df['USD_INR'].shift(3)

df['MA_3'] = df['USD_INR'].rolling(3).mean()
df['MA_7'] = df['USD_INR'].rolling(7).mean()

# Target
df['Target'] = df['USD_INR'].shift(-1)

df = df.dropna()

X = df[['Lag_1','Lag_2','Lag_3','MA_3','MA_7']]
y = df['Target']

# Train
model = Ridge()
model.fit(X, y)

# Save
import os
os.makedirs("models", exist_ok=True)

with open("models/usd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("USD model trained & saved")