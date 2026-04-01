import pandas as pd
import yfinance as yf
import os
import numpy as np

# FILE PATHS
GOLD_FILE = "data/raw/gold_raw.csv"
SILVER_FILE = "data/raw/silver_raw.csv"
OUTPUT_FILE = "data/processed/final_data.csv"

OUNCE_TO_GRAM = 31.1035



# LOAD DATA

def load_data():
    gold = pd.read_csv(GOLD_FILE, low_memory=False)
    silver = pd.read_csv(SILVER_FILE, low_memory=False)

    # Convert Date
    gold['Date'] = pd.to_datetime(gold['Date'], errors='coerce')
    silver['Date'] = pd.to_datetime(silver['Date'], errors='coerce')

    # Convert numeric
    for col in gold.columns:
        if col != "Date":
            gold[col] = pd.to_numeric(gold[col], errors='coerce')

    for col in silver.columns:
        if col != "Date":
            silver[col] = pd.to_numeric(silver[col], errors='coerce')

    gold = gold.dropna(subset=['Date', 'Close'])
    silver = silver.dropna(subset=['Date', 'Close'])

    return gold, silver



# FETCH USD/INR

def fetch_usd_inr():
    print("Fetching USD/INR data...")

    usd = yf.download("USDINR=X", start="2015-01-01")

    if isinstance(usd.columns, pd.MultiIndex):
        usd.columns = usd.columns.get_level_values(0)

    usd.reset_index(inplace=True)
    usd = usd[['Date', 'Close']]
    usd.rename(columns={'Close': 'USD_INR'}, inplace=True)

    return usd



# PREPROCESS

def preprocess():
    gold, silver = load_data()

    # Select required columns
    gold = gold[['Date', 'Close']]
    silver = silver[['Date', 'Close']]

    # Rename
    gold.rename(columns={'Close': 'Gold_USD'}, inplace=True)
    silver.rename(columns={'Close': 'Silver_USD'}, inplace=True)

    # Merge Gold + Silver
    df = pd.merge(gold, silver, on='Date', how='outer')

    # Fetch USD
    usd = fetch_usd_inr()
    usd['Date'] = pd.to_datetime(usd['Date'])

    df = pd.merge(df, usd, on='Date', how='left')

    
    #  CLEAN DATE PROPERLY (NO ERROR ZONE)
    
    df = df.loc[:, ~df.columns.duplicated()]   # remove duplicate columns
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df = df.sort_values('Date')
    df = df.drop_duplicates(subset='Date')

    
    # RESAMPLE (NO reset_index issue)
    
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill()

    # bring Date back safely
    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    
    # FILL VALUES
    
    df['USD_INR'] = df['USD_INR'].ffill()
    df = df.ffill().bfill()

    
    # PRICE CONVERSION
    
    df['Gold_24K_1g'] = (df['Gold_USD'] / OUNCE_TO_GRAM) * df['USD_INR']
    df['Silver_1g'] = (df['Silver_USD'] / OUNCE_TO_GRAM) * df['USD_INR']

    # RETURNS
    df['Gold_Return'] = df['Gold_24K_1g'].pct_change()
    df['Silver_Return'] = df['Silver_1g'].pct_change()

    # VOLATILITY
    df['Gold_Volatility'] = df['Gold_Return'].rolling(7).std()

    # USD CHANGE
    df['USD_Change'] = df['USD_INR'].pct_change()

    # RATIO
    df['Gold_Silver_Ratio'] = df['Gold_24K_1g'] / df['Silver_1g']

    # Gold 22K
    df['Gold_22K_1g'] = df['Gold_24K_1g'] * (22 / 24)

    
    # WEIGHTS
    
    weights = {"1g": 1, "10g": 10, "100g": 100, "1kg": 1000}
    metals = ["Gold_24K", "Gold_22K", "Silver"]

    for metal in metals:
        for w, val in weights.items():
            df[f"{metal}_{w}"] = df[f"{metal}_1g"] * val

    
    # CLEAN FINAL
    
    df.drop(columns=['Gold_USD', 'Silver_USD'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.dropna(subset=['Gold_24K_1g', 'Silver_1g', 'USD_INR'])

    df = df.sort_values('Date').reset_index(drop=True)

    print("\nFinal Data Shape:", df.shape)
    print("Missing Values:\n", df.isnull().sum())

    # SAVE
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n Final data saved →", OUTPUT_FILE)

    return df



# RUN

if __name__ == "__main__":
    print("Processing Data...\n")
    preprocess()
    print("\nDone ")