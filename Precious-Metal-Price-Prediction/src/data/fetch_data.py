import yfinance as yf
import pandas as pd
import os

# FILE PATHS

GOLD_FILE = "data/raw/gold_raw.csv"
SILVER_FILE = "data/raw/silver_raw.csv"

# SYMBOLS
GOLD_SYMBOL = "GC=F"
SILVER_SYMBOL = "SI=F"

# SAFE DOWNLOAD FUNCTION
def safe_download(symbol):
    df = yf.download(symbol, period="7d", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    # CASE 1: MultiIndex columns (VERY COMMON BUG)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # RESET INDEX (DATE COMES HERE)
    df = df.reset_index()

    # ENSURE DATE COLUMN EXISTS
    if 'Date' not in df.columns:
        # Sometimes index name is None → first column is date
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # CLEAN DATE
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    return df

# CORE FUNCTION
def fetch_data(symbol, file_path):
    os.makedirs("data/raw", exist_ok=True)

    print(f"\nFetching {symbol}...")

    try:
        new_data = safe_download(symbol)

        print("New data fetched")
        print("Latest new date:", new_data['Date'].max().date())

    except Exception as e:
        print(f"Fetch failed: {e}")
        return pd.DataFrame()

    # MERGE WITH OLD DATA
    if os.path.exists(file_path):
        try:
            old_data = pd.read_csv(file_path, low_memory=False)

            if not old_data.empty:
                old_data['Date'] = pd.to_datetime(old_data['Date'], errors='coerce')
                old_data = old_data.dropna(subset=['Date'])

                # MERGE
                df = pd.concat([old_data, new_data])

                # Remove duplicates
                df = df.drop_duplicates(subset='Date')

                # Sort
                df = df.sort_values('Date').reset_index(drop=True)

            else:
                df = new_data

        except Exception as e:
            print(f"Old file corrupted → rebuilding: {e}")
            df = new_data

    else:
        print("First-time download")
        df = new_data

    
    # FINAL CHECK
    
    if df.empty:
        raise ValueError(f"Final dataset empty for {symbol}")

    # SAVE
    df.to_csv(file_path, index=False)

    print(f"Saved → {file_path}")
    print(f"Final latest date → {df['Date'].max().date()}")

    return df

# WRAPPERS

def fetch_gold_data():
    return fetch_data(GOLD_SYMBOL, GOLD_FILE)


def fetch_silver_data():
    return fetch_data(SILVER_SYMBOL, SILVER_FILE)


def fetch_all():
    print("\nFetching All Market Data...\n")

    gold = fetch_gold_data()
    silver = fetch_silver_data()

    print("\nAll Data Updated!\n")

    return gold, silver

# MAIN

if __name__ == "__main__":
    print("Starting Data Fetch...\n")
    fetch_all()
    print("\nDone!")