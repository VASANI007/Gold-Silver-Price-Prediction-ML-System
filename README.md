<!-- 🌌 Header --> 
<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=220&section=header&text=Precious%20Metal%20Price%20Prediction&fontSize=40&fontColor=ffffff&animation=fadeIn"/>
</p>

---

# 🪙 Precious Metal Price Prediction & Market Analytics

An advanced **Machine Learning project** that predicts gold prices and analyzes precious metal trends using real-time financial data.

---

# 🏆 Tech Stack

![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)

---

# 📌 Project Overview

This project provides a **complete ML pipeline + dashboard** for:

- 📡 Real-time Gold & Silver data collection  
- 💱 USD → INR price conversion  
- 📊 Data preprocessing & feature engineering  
- 🧠 Machine learning model training  
- 🔮 Future price prediction  
- 📈 Interactive dashboard visualization  

---

# 📊 Model Performance

| Metric   | 🪙 Gold | 🔘 Silver | 💱 USD |
|----------|--------|----------|--------|
| MAE      | ₹31.80 | ₹0.82    | ₹0.14  |
| RMSE     | ₹54.78 | ₹1.81    | ₹0.23  |
| R² Score | 0.9826 | 0.9663   | 0.9831 |

> 🎯 **High accuracy achieved across all models with stable predictions**

---

# 📂 Project Structure

```
PRECIOUS-METAL-PRICE-PREDICTION/

├── app/
│   └── app.py                     # Streamlit Dashboard
│
├── data/
│   ├── raw/
│   │   ├── gold_raw.csv
│   │   └── silver_raw.csv
│   │
│   └── processed/
│       └── final_data.csv
│
├── images/                       # App screenshots (optional but recommended)
│
├── models/
│   ├── model.pkl                 
│   ├── silver_model.pkl          
│   ├── usd_model.pkl            
│   │
│   ├── gold_metrics.pkl          
│   ├── silver_metrics.pkl        
│   └── usd_metrics.pkl           
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── data/
│   │   └── fetch_data.py
│   │
│   ├── processing/
│   │   └── preprocess.py
│   │
│   └── models/
│       ├── train_gold_model.py   
│       ├── train_silver_model.py  
│       ├── train_usd_model.py     
│       └── predict.py
│
├── main.py                        # Full pipeline runner
├── requirements.txt
├── README.md                      # IMPORTANT
```

---

# 🪙 Precious Metal Price Prediction & Market Analytics

An advanced **Machine Learning + Data Analytics project** that predicts gold prices and analyzes precious metals using real-time financial data.

---

# 🚀 Key Highlights

- 📡 Real-time data pipeline (Gold, Silver, USD-INR)
- 💱 Dynamic USD → INR conversion
- 🧠 Machine Learning-based prediction system
- 🔮 7-day future forecasting
- 📊 Interactive Streamlit dashboard
- ⚡ Automated daily updates

---

# 🧠 How It Works

1. Fetch real-time data using Yahoo Finance  
2. Merge Gold, Silver & USD-INR  
3. Convert prices into INR per gram  
4. Feature Engineering:
   - Lag values  
   - Moving averages  
   - Volatility  
   - Currency impact  
5. Train ML model  
6. Predict future prices  
7. Visualize in dashboard  

---

# 🖼️ Dashboard Preview

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Gold-Silver-Price-Prediction-ML-System/main/Precious-Metal-Price-Prediction/images/app.png" width="45%"/>
</p>

---

# 📊 Analysis & Insights (Notebook)

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Gold-Silver-Price-Prediction-ML-System/main/Precious-Metal-Price-Prediction/images/gold_24_notebook.png" width="45%"/>
<img src="https://raw.githubusercontent.com/VASANI007/Gold-Silver-Price-Prediction-ML-System/main/Precious-Metal-Price-Prediction/images/gold_vs_silver_notbook.png" width="45%"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Gold-Silver-Price-Prediction-ML-System/main/Precious-Metal-Price-Prediction/images/silver_notebook.png" width="45%"/>
<img src="https://raw.githubusercontent.com/VASANI007/Gold-Silver-Price-Prediction-ML-System/main/Precious-Metal-Price-Prediction/images/year_gold_notebook.png" width="45%"/>
</p>

---

# 🏗️ Project Architecture

Data Fetch → Preprocessing → Feature Engineering → Model Training → Prediction → Dashboard

---

# 📄 File Explanation (Important Section 🚀)

### 🔹 app/app.py
- Streamlit dashboard UI  
- Shows charts, predictions, and analytics  
- Handles user interaction  

---

### 🔹 data/raw/
- Contains original dataset  
- `gold_raw.csv` → Gold price data  
- `silver_raw.csv` → Silver price data  

---

### 🔹 data/processed/final_data.csv
- Final cleaned dataset  
- Includes INR conversion + engineered features  
- Used for model training & dashboard  

---

### 🔹 models/
- Stores trained models and evaluation metrics  

- `model.pkl` → Gold model  
- `silver_model.pkl` → Silver model  
- `usd_model.pkl` → USD model  

- `gold_metrics.pkl` → Gold model performance (MAE, RMSE, R²)  
- `silver_metrics.pkl` → Silver model performance  
- `usd_metrics.pkl` → USD model performance  

---

### 🔹 notebooks/analysis.ipynb
- Exploratory Data Analysis (EDA)  
- Visualization and experimentation  
- Model understanding  

---

### 🔹 src/data/fetch_data.py
- Fetches real-time data using Yahoo Finance API  
- Updates dataset automatically  

---

### 🔹 src/processing/preprocess.py
- Cleans and merges datasets  
- Converts USD → INR  
- Creates new features  
- Generates final dataset  

---

### 🔹 src/models/train_model.py
- Trains Gold prediction model  
- Uses processed dataset  
- Saves model as `model.pkl`  

---

### 🔹 src/models/train_silver_model.py
- Trains Silver prediction model  
- Uses lag features, moving averages & gold influence  
- Saves model as `silver_model.pkl`  
- Also saves performance metrics  

---

### 🔹 src/models/train_usd_model.py
- Trains USD-INR prediction model  
- Uses lag & moving average features  
- Saves model as `usd_model.pkl`  
- Stores evaluation metrics  

---

### 🔹 src/models/predict.py
- Loads trained models  
- Performs predictions for Gold, Silver, and USD  
- Integrated with dashboard  

---

### 🔹 main.py
- Runs full pipeline automatically  
- Data → Processing → Training → Prediction  
- One command execution  

---

### 🔹 requirements.txt
- List of all dependencies  
- Used for environment setup  

---

# ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

### 🔹 Run Full Pipeline
```bash
python main.py
```

### 🔹 Run Dashboard
```bash
streamlit run app/app.py
```

---

# 📊 Dashboard Features

- 📈 Price trend graphs  
- 📊 Multi-weight pricing (1g → 1kg)  
- 📉 Daily change indicators  
- 💰 Gold (24K / 22K) & Silver analysis  
- 💱 USD-INR tracking  
- 🔄 Auto refresh system  

---

# 🔮 Prediction System

- Next day gold price prediction  
- 7-day future forecasting  
- Based on time + moving average features  

---

# 🚀 Future Improvements

- Deep Learning models  
- Cloud deployment  
- Real-time streaming  
- Crypto integration  

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=170&section=footer&text=Thanks%20for%20Visiting%20My%20Profile!&fontSize=28&fontColor=ffffff&animation=twinkling&fontAlignY=65"/>
</p>
