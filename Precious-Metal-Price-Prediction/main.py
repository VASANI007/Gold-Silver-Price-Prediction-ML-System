import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback
from src.data.fetch_data import fetch_all
from src.processing.preprocess import preprocess
from src.models.train_model import train_model
from src.models.train_usd_model import train_usd_model
from src.models.train_silver_model import train_silver_model
from src.models.predict import *


def run_pipeline():
    print("\n Starting Full Pipeline...\n")

    try:
        # STEP 1: Fetch Data
        print(" Fetching Data...")
        fetch_all()
        print(" Data Fetch Complete\n")

        # STEP 2: Preprocess
        print(" Processing Data...")
        preprocess()
        print(" Data Processing Complete\n")

        # STEP 3: Train Model
        print(" Training Model...")
        train_model()
        train_usd_model()
        train_silver_model()
        print(" Model Training Complete\n")

        # STEP 4: Predictions
        print(" Running Predictions...")

        gold, silver, usd = predict_all()

        print(f"\n Gold Prediction: ₹ {gold:.2f}")
        print(f" Silver Prediction: ₹ {silver:.2f}")
        print(f" USD Prediction: ₹ {usd:.2f}")

        future = predict_future(7)
        print("\n Next 7 Days Forecast:")
        for i, val in enumerate(future, start=1):
            print(f"Day {i}: ₹ {val:.2f}")

        print("\n Prediction Complete\n")

    except Exception as e:
        print("\n Pipeline Failed!")
        print(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    run_pipeline()
