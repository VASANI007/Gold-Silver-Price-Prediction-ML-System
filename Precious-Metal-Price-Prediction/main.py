import traceback
from src.data.fetch_data import fetch_all
from src.processing.preprocess import preprocess
from src.models.train_model import train_model
from src.models.train_usd_model import train_usd_model
from src.models.predict import predict_next, predict_future


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
        print(" Model Training Complete\n")

        # STEP 4: Predictions
        print(" Running Predictions...")

        next_day = predict_next()
        print(f"\n Next Day Prediction: ₹ {next_day:.2f}")

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
