import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from src.data.fetch_data import fetch_all
from src.processing.preprocess import preprocess

#  PATH FIX

#  CONFIG
st.set_page_config(page_title="Gold & Silver Market Insights", page_icon="🪙", layout="wide")
#  STYLES Title
st.markdown("""
<h1 style='
    color:white;
    border-left:6px solid #888;
    padding-left:12px;
    font-weight:bold;
'>
Gold & Silver Market Insights
</h1>
<p style='color:#aaa; margin-left:12px;'>
Advanced Analytics for Gold, Silver & Currency
</p>
""", unsafe_allow_html=True)

# ADVANCED LOADING SCREEN
loading_placeholder = st.empty()

loading_placeholder.markdown("""
<style>
.loader-container {
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    height:60vh;
    text-align:center;
}

.loader {
    border: 6px solid #1a1a1a;
    border-top: 6px solid #4FC3F7;
    border-radius: 50%;
    width: 70px;
    height: 70px;
    animation: spin 1s linear infinite;
    margin-bottom:20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-text {
    font-size:20px;
    color:white;
    font-weight:500;
    text-shadow: 0 0 10px #4FC3F7;
}

.loading-sub {
    font-size:14px;
    color:#888;
    margin-top:5px;
}
</style>

<div class="loader-container">
    <div class="loader"></div>
    <div class="loading-text">Loading Market Intelligence...</div>
    <div class="loading-sub">Fetching Gold, Silver & Currency Data</div>
</div>
""", unsafe_allow_html=True)

#  STYLES TABLE
st.markdown("""
<style>
table {
    width: 100% !important;
    border-collapse: collapse;
    text-align: center;
    font-size: 16px;
}
th {
    background-color: #111;
    color: white;
    padding: 14px;
    text-align: center !important;
}
td {
    padding: 12px;
    border-bottom: 1px solid #333;
    text-align: center !important;
}
tr:hover {
    background-color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

#  STYLES SUBHEADER
def styled_subheader(text):
    st.markdown(f"""
    <h3 style='
        border-left: 5px solid #4FC3F7;
        padding-left: 10px;
        font-weight: 400;
        margin-top: 20px;
    '>
    {text}
    </h3><br>
    """, unsafe_allow_html=True)

#  AUTO REFRESH
with st.spinner(" Fetching market data..."):
    try:
        fetch_all()
        preprocess()
    except Exception as e:
        st.warning(f"Data update skipped: {e}")


#  LOAD DATA
def load_data():
    try:
        df = pd.read_csv("data/processed/final_data.csv")

        if df.empty:
            raise ValueError("Empty dataset")

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        return df

    except Exception as e:
        st.error("Failed to load data. Please update data.")
        return pd.DataFrame()
# LOAD MODEL
try:
    model = joblib.load("models/model.pkl")
    usd_model = joblib.load("models/usd_model.pkl")
    silver_model = joblib.load("models/silver_model.pkl")
    gold_metrics = joblib.load("models/gold_metrics.pkl")
    silver_metrics = joblib.load("models/silver_metrics.pkl")
    usd_metrics = joblib.load("models/usd_metrics.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()
df = load_data()
loading_placeholder.empty()
if not df.empty:
    st.caption(f"Latest available data: {df['Date'].max().date()}")
if df.empty or len(df) < 2:
    st.error("Not enough data available")
    st.stop()

latest = df.iloc[-1]
previous = df.iloc[-2]

today_date = latest['Date'].date()
yesterday_date = previous['Date'].date()

#  SCROLLING TICKER
@st.cache_data(ttl=300)
def get_usd_data():
    return yf.download("USDINR=X", period="2d")
#  LOAD USD DATA
@st.cache_data(ttl=300)
def load_usd_full():
    usd = yf.download("USDINR=X", period="1y")
    if isinstance(usd.columns, pd.MultiIndex):
        usd.columns = usd.columns.get_level_values(0)
    usd.reset_index(inplace=True)
    return usd

usd = load_usd_full()

#  CALCULATE CHANGES
g24_change = latest['Gold_24K_1g'] - previous['Gold_24K_1g']
g22_change = latest['Gold_22K_1g'] - previous['Gold_22K_1g']
silver_change = latest['Silver_1g'] - previous['Silver_1g']
try:
    usd_live = usd.tail(2)
    usd_live = usd_live.dropna()
    if len(usd_live) < 2:
        raise ValueError("Not enough live data")
    usd_price = float(usd_live['Close'].iloc[-1].item())
    usd_prev = float(usd_live['Close'].iloc[-2].item())
except Exception as e:
    usd_price = float(usd['Close'].iloc[-1])
    usd_prev = float(usd['Close'].iloc[-2])


usd_change = usd_price - usd_prev
#  SCROLLING TICKER
def format_change(val):
    if val > 0:
        return f"<span style='color:#02ff99; font-weight:bold;'>▲ {abs(val):.2f}</span>"
    elif val < 0:
        return f"<span style='color:#ff4d4d; font-weight:bold;'>▼ {abs(val):.2f}</span>"
    else:
        return "<span style='color:gray;'>0</span>"

def predict_usd_next(latest, prev, usd):

    lag1 = latest
    lag2 = prev
    lag3 = usd['Close'].iloc[-3]

    ma3 = usd['Close'].tail(3).mean()
    ma7 = usd['Close'].tail(7).mean()

    X = pd.DataFrame([[lag1, lag2, lag3, ma3, ma7]],
                        columns=['Lag_1','Lag_2','Lag_3','MA_3','MA_7'])

    return usd_model.predict(X)[0]

# ADD THIS GLOBAL FUNCTION (TOP ME ADD KARO)
def create_input(lag1, lag2, lag3, ma7, ma30, day_of_week):

    usd_change = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    den = df['Silver_1g'].iloc[-2]
    silver_change = ((df['Silver_1g'].iloc[-1] - den) / den) if den != 0 else 0

    input_data = pd.DataFrame([{
        "Lag_1": lag1,
        "Lag_2": lag2,
        "Lag_3": lag3,
        "MA_7": ma7,
        "MA_30": ma30,
        "USD_INR": df['USD_INR'].iloc[-1],
        "USD_Change": usd_change,
        "Silver_1g": df['Silver_1g'].iloc[-1],
        "Silver_Change": silver_change,
        "Gold_22K_1g": df['Gold_22K_1g'].iloc[-1],
        "DayOfWeek": day_of_week
    }])

    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    return input_data[model.feature_names_in_]

#  PREPARE VALUES
g24_html = format_change(g24_change)
g22_html = format_change(g22_change)
silver_html = format_change(silver_change)
usd_html = format_change(usd_change)
#  SINGLE LINE STRING (IMPORTANT)
ticker_text = f"""
Gold 24K: ₹ {latest['Gold_24K_1g']:.2f}&nbsp;&nbsp;&nbsp;&nbsp; ({g24_html})
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Gold 22K: ₹ {latest['Gold_22K_1g']:.2f}&nbsp;&nbsp;&nbsp;&nbsp; ({g22_html})
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Silver: ₹ {latest['Silver_1g']:.2f}&nbsp;&nbsp;&nbsp;&nbsp; ({silver_html})
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
USD/INR: ₹ {usd_price:.2f}&nbsp;&nbsp;&nbsp;&nbsp; ({usd_html})
""".replace("\n", " ")

#  RENDER
st.markdown(f"""
<style>
.ticker-container {{
    width: 100%;
    overflow: hidden;
    background: #0e1117;
    padding: 10px 0;
}}

.ticker-text {{
    display: inline-block;
    white-space: nowrap;
    animation: scroll-left 12s linear infinite;
    color: white;
    font-size: 17px;
    padding-left: 100%;
}}

@keyframes scroll-left {{
    0% {{ transform: translateX(0%); }}
    100% {{ transform: translateX(-100%); }}
}}
</style>

<div class="ticker-container">
    <div class="ticker-text">
        {ticker_text}
    </div>
</div>
""", unsafe_allow_html=True)
#  TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🪙 Gold 24K",
    "🧈 Gold 22K",
    "🔘 Silver",
    "💱 USD → INR",
    "📊 Model Performance"
])

def create_silver_input(df, lag1, lag2, lag3, ma7, ma30):

    # safe calculations
    return_val = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    momentum = lag1 - lag2

    ema_10 = df['Silver_1g'].ewm(span=10).mean().iloc[-1]
    rolling_std = df['Silver_1g'].rolling(5).std().iloc[-1]

    input_data = pd.DataFrame([{
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

    return input_data

#  COMMON UI
def show_section(metal, column_name):

    styled_subheader(f"{metal} Overview")

    #  DATE + WEIGHT 
    colA, colB = st.columns(2)
    max_date = df['Date'].max().date()
    with colA:

        min_date = df['Date'].min().date()              # past unlimited
        today = datetime.now().date()
        max_future_date = today + timedelta(days=7)     # future limit

        selected_date = st.date_input(
                                    "Choose Date",
                                    value=df['Date'].max().date(),
                                    min_value=min_date,          # past allowed
                                    max_value=max_future_date,   # future only 7 days
                                    key=f"{metal}_date"
                                    )
        if selected_date > max_future_date:
            st.error("Only next 7 days allowed for prediction")
            return

    with colB:
        weight_options = ["1g", "10g", "100g", "1kg"]
        selected_weight = st.selectbox(
            "Select Weight",
            weight_options,
            index=0,
            key=f"{metal}_weight"
        )

    # IMPORTANT: dynamic column
    column_name = f"{metal}_{selected_weight}"

    filtered_df = df[df['Date'].dt.date == selected_date]

    max_date = df['Date'].max().date()

    #  CASE 1: FUTURE DATE
    if filtered_df.empty and selected_date > max_date:
        st.warning("Future date selected — showing prediction")

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        last_val = float(last_row[column_name])
        prev_val = float(prev_row[column_name])

        days_ahead = (selected_date - max_date).days

        temp_prev = prev_val
        temp_last = last_val
        day_of_week = pd.to_datetime(selected_date).dayofweek
        temp_prices = df[column_name].tolist()
        for _ in range(days_ahead):
            lag1 = temp_last
            lag2 = temp_prev

            lag3 = temp_prices[-3] if len(temp_prices) >= 3 else lag2

            ma7 = df[column_name].rolling(7).mean().iloc[-1] if len(df) >= 7 else lag1
            ma30 = df[column_name].rolling(30).mean().iloc[-1] if len(df) >= 30 else lag1

            input_data = create_input(lag1, lag2, lag3, ma7, ma30, day_of_week)
            gold_pred = model.predict(input_data)[0]

            if metal == "Silver":
                X = create_silver_input(df, lag1, lag2, lag3, ma7, ma30)
                next_pred = silver_model.predict(X)[0]
                next_pred = (next_pred * 0.7) + (lag1 * 0.3)
                mean_price = df[column_name].tail(7).mean()
                next_pred = (next_pred * 0.6 + lag1 * 0.2 + mean_price * 0.2)
            elif metal == "Gold_22K":
                next_pred = gold_pred * (22/24)
            else:
                next_pred = gold_pred

            temp_prev = temp_last
            temp_last = next_pred
            if metal == "Silver":
                temp_last = (temp_last * 0.7) + (lag1 * 0.3)
            

        today = temp_last
        yesterday = temp_prev

        selected_datetime = pd.to_datetime(selected_date)

    #  IMPORTANT FIX (TABLE KE LIYE)
        selected_row = {}

        for w in ["1g", "10g", "100g", "1kg"]:
            col = f"{metal}_{w}"

            last_val_w = float(df.iloc[-1][col])
            prev_val_w = float(df.iloc[-2][col])

            temp_prev_w = prev_val_w
            temp_last_w = last_val_w
            temp_prices = df[col].tolist()
            day_of_week = selected_datetime.dayofweek
            for _ in range(days_ahead):
                lag1 = temp_prices[-1]
                lag2 = temp_prices[-2]
                lag3 = temp_prices[-3] if len(temp_prices) >= 3 else lag2
                ma7 = np.mean(temp_prices[-7:]) if len(temp_prices) >= 7 else lag1
                ma30 = np.mean(temp_prices[-30:]) if len(temp_prices) >= 30 else lag1

                input_data = create_input(lag1, lag2, lag3, ma7, ma30, day_of_week)
                gold_pred = model.predict(input_data)[0]

                if metal == "Silver":
                    X = create_silver_input(df, lag1, lag2, lag3, ma7, ma30)
                    next_pred = silver_model.predict(X)[0]
                    next_pred = (next_pred * 0.7) + (lag1 * 0.3)
                    mean_price = df[column_name].tail(7).mean()
                    next_pred = (next_pred * 0.6 + lag1 * 0.2 + mean_price * 0.2)
                elif metal == "Gold_22K":
                    next_pred = gold_pred * (22/24)
                else:
                    next_pred = gold_pred

                temp_prices.append(next_pred)

            selected_row[col] = temp_prices[-1]

    #  CASE 2: NORMAL DATE
    elif not filtered_df.empty:
        selected_row = filtered_df.iloc[0]

        if metal == "Gold_22K":
            today = float(selected_row["Gold_24K_1g"]) * (22/24)
        elif metal == "Silver":
            today = float(selected_row["Silver_1g"])
        else:
            today = float(selected_row[column_name])

        prev_df = df[df['Date'] < pd.to_datetime(selected_date)]

        if not prev_df.empty:
            yesterday = float(prev_df.iloc[-1][column_name])
        else:
            yesterday = today

        selected_datetime = pd.to_datetime(selected_date)

#  CASE 3: INVALID
    else:
        st.error("Data Not Found for selected date")
        return

    prev_df = df[df['Date'] < pd.to_datetime(selected_date)]

    if not prev_df.empty:
        yesterday = prev_df.iloc[-1][column_name]
    else:
        yesterday = today

    change = today - yesterday

    if change > 0:
        arrow = "▲"
        delta_color = "normal"
    else:
        arrow = "▼"
        delta_color = "inverse"

    #  METRICS 
    col1, col2, col3, col4 = st.columns(4)
    selected_datetime = pd.to_datetime(selected_date)
    #  PREDICTION
    if model is not None:
        try:

            last_value = float(today)
            prev_value = float(yesterday)

            lag1 = last_value
            lag2 = prev_value
            temp_prices = df[column_name].tolist()

            lag3 = temp_prices[-3] if len(temp_prices) >= 3 else lag2
            day_of_week = selected_datetime.dayofweek
            ma7 = df[column_name].rolling(7).mean().iloc[-1] if len(df) >= 7 else lag1
            ma30 = df[column_name].rolling(30).mean().iloc[-1] if len(df) >= 30 else lag1
            input_data = create_input(lag1, lag2, lag3, ma7, ma30, day_of_week)
            gold_pred = model.predict(input_data)[0]

            if metal == "Silver":
                X = create_silver_input(df, lag1, lag2, lag3, ma7, ma30)
                prediction = silver_model.predict(X)[0]
            elif metal == "Gold_22K":
                prediction = gold_pred * (22/24)
            else:
                prediction = gold_pred

            pred_change = prediction - last_value

            arrow = "▲" if pred_change > 0 else "▼"
            color = "normal" if pred_change > 0 else "inverse"

            st.metric(
                "🎯 Predicted Next Day",
                f"₹ {prediction:.2f}",
                f"{arrow} {abs(pred_change):.2f}",
                delta_color=color
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning(" Model not loaded")
    #  FUTURE 7 DAYS PREDICTION
    future_days = 7
    future_preds = []
    future_dates = []

    last_val = float(today)
    prev_val = float(yesterday)
    day_of_week = selected_datetime.dayofweek
    temp_prices = df[column_name].tolist()
    for i in range(future_days):
        lag1 = temp_prices[-1]
        lag2 = temp_prices[-2]
        lag3 = temp_prices[-3] if len(temp_prices) >= 3 else lag2

        ma7 = np.mean(temp_prices[-7:]) if len(temp_prices) >= 7 else lag1
        ma30 = np.mean(temp_prices[-30:]) if len(temp_prices) >= 30 else lag1
        input_data = create_input(lag1, lag2, lag3, ma7, ma30, day_of_week)
        gold_pred = model.predict(input_data)[0]

        if metal == "Silver":
            X = create_silver_input(df, lag1, lag2, lag3, ma7, ma30)
            next_pred = silver_model.predict(X)[0]
            next_pred = (next_pred * 0.7) + (lag1 * 0.3)
            mean_price = df[column_name].tail(7).mean()
            next_pred = (next_pred * 0.6 + lag1 * 0.2 + mean_price * 0.2)
        elif metal == "Gold_22K":
             next_pred = gold_pred * (22/24)
        else:
            next_pred = gold_pred

        future_preds.append(next_pred)
        temp_prices.append(next_pred)

        prev_val = last_val
        last_val = next_pred
        future_dates.append(selected_datetime + timedelta(days=i+1))

    col1.metric("📅 Selected Day", f"₹ {today:.2f}", f"{arrow} {change:.2f}", delta_color=delta_color)
    col2.metric("🗓️ Previous Day", f"₹ {yesterday:.2f}")
    col3.metric("📈 Highest", f"₹ {df[column_name].max():.2f}")
    col4.metric("📉 Lowest", f"₹ {df[column_name].min():.2f}")

    st.caption(f"Selected Date: {selected_date} | Weight: {selected_weight}")

    #  TABLE 
    styled_subheader("Price Table")

    weights = ["1g", "10g", "100g", "1kg"]
    rows = []

    for w in weights:
        col = f"{metal}_{w}"
        t = selected_row[col] if isinstance(selected_row, dict) else selected_row[col]

        if not prev_df.empty:
            y = prev_df.iloc[-1][col]
        else:
            y = t

        c = t - y

        if c > 0:
            change_html = f"<span style='color:#02ff99'>▲ ₹{abs(c):,.2f}</span>"
        else:
            change_html = f"<span style='color:#ff4d4d'>▼ ₹{abs(c):,.2f}</span>"

        rows.append({
            "Gram": w,
            "Today": f"₹{t:,.2f}",
            "Yesterday": f"₹{y:,.2f}",
            "Change": change_html
        })

    table_df = pd.DataFrame(rows)

    st.markdown(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    styled_subheader(" 7 Day Prediction")

    pred_rows = []

    for i in range(len(future_preds)):
        if i == 0:
            prev_val = today
        else:
            prev_val = future_preds[i-1]

        curr = future_preds[i]
        diff = curr - prev_val

        if diff > 0:
            change_html = f"<span style='color:#02ff99'>▲ ₹{abs(diff):,.2f}</span>"
        else:
            change_html = f"<span style='color:#ff4d4d'>▼ ₹{abs(diff):,.2f}</span>"

        pred_rows.append({
            "Date": future_dates[i].date(),
            "Predicted Price": f"₹{curr:,.2f}",
            "Change": change_html
        })

    pred_df = pd.DataFrame(pred_rows)

    st.markdown(pred_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    #  GRAPH 
    styled_subheader("Price Trend")

    options = ["1W", "1M", "3M", "6M", "1Y", "ALL"]

    key_name = f"{metal}_range"

    if key_name not in st.session_state:
        st.session_state[key_name] = "3M"

    left, c1, c2, c3, c4, c5, c6, right = st.columns([2,1,1,1,1,1,1,2])
    cols = [c1, c2, c3, c4, c5, c6]

    for i, opt in enumerate(options):
        if cols[i].button(opt, key=f"{opt}_{metal}"):
            st.session_state[key_name] = opt

    selected = st.session_state[key_name]
    selected_datetime = pd.to_datetime(selected_date)

    if selected == "1W":
        dff = df[df['Date'] <= selected_datetime].tail(7)
    elif selected == "1M":
        dff = df[df['Date'] <= selected_datetime].tail(30)
    elif selected == "3M":
        dff = df[df['Date'] <= selected_datetime].tail(90)
    elif selected == "6M":
        dff = df[df['Date'] <= selected_datetime].tail(180)
    elif selected == "1Y":
        dff = df[df['Date'] <= selected_datetime].tail(365)
    else:
        dff = df[df['Date'] <= selected_datetime]

    if metal == "Gold_22K":
        dff = dff.copy()
        dff[column_name] = dff["Gold_24K_1g"] * (22/24)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=dff['Date'],
    y=dff[column_name],
    mode='lines',
    line=dict(color='#2ecc71', width=3),
    name=f"{selected_weight} Price"
))

    fig.add_trace(go.Scatter(
    x=dff['Date'],
    y=dff[column_name],
    fill='tozeroy',
    mode='none',
    fillcolor='rgba(46,204,113,0.1)'
))

    fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    yaxis_title=f"Price ({selected_weight})"
)


#  INCLUDE PREDICTION IN RANGE
    all_values = list(dff[column_name].values) + list(future_preds)

    y_min = min(all_values)
    y_max = max(all_values)

    fig.update_yaxes(range=[y_min * 0.95, y_max * 1.05])

# ADD PREDICTION LINE
    fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_preds,
    mode='lines+markers',
    line=dict(color='#f39c12', width=3, dash='dash'),
    name="Prediction"
))

    st.plotly_chart(fig, width='stretch')
#  TABS CONTENT
with tab1:
    show_section("Gold_24K", "Gold_24K_1g")

with tab2:
    show_section("Gold_22K", "Gold_22K_1g")

with tab3:
    show_section("Silver", "Silver_1g")

#  USD TAB
with tab4:
    styled_subheader(" USD to INR")

    try:
        usd = yf.download("USDINR=X", period="1y")

        if usd.empty:
            st.error("No USD-INR data available")
            st.stop()

        if isinstance(usd.columns, pd.MultiIndex):
            usd.columns = usd.columns.get_level_values(0)

        usd.reset_index(inplace=True)
        usd = usd[['Date', 'Close']].dropna()
        usd['Date'] = pd.to_datetime(usd['Date'])

        # DATE PICKER
        min_date = usd['Date'].min().date()              # past unlimited
        today = datetime.now().date()
        max_future_date = today + timedelta(days=7)      # future limit

        selected_date = st.date_input(
                                    "Choose Date",
                                    value=usd['Date'].iloc[-1].date(),
                                    min_value=min_date,
                                    max_value=max_future_date,
                                    key="usd_date"
        )
        if selected_date > max_future_date:
            st.error("Only next 7 days allowed for prediction")
            st.stop()

        filtered_usd = usd[usd['Date'].dt.date == selected_date]
        max_date = usd['Date'].max().date()
        today = pd.to_datetime("today").normalize()
        selected_datetime = pd.to_datetime(selected_date)
        # FUTURE DATE
        if selected_datetime > today:
            st.warning("Future Prediction Mode")

            last_row = usd.iloc[-1]
            prev_row = usd.iloc[-2]

            last_val = float(last_row['Close'])
            prev_val = float(prev_row['Close'])

            days_ahead = (selected_datetime - usd['Date'].max()).days

            temp_prev = prev_val
            temp_last = last_val

            for _ in range(days_ahead):
                next_pred = temp_last  # (your logic)
                temp_prev = temp_last
                temp_last = next_pred

            latest = temp_last
            prev = temp_prev


# TODAY (LIVE / LATEST)
        elif selected_datetime == today:

            last_row = usd.iloc[-1]
            prev_row = usd.iloc[-2]

            latest = float(last_row['Close'])
            prev = float(prev_row['Close'])


# PAST DATE
        elif not filtered_usd.empty:
            st.info("Historical Data")

            selected_row = filtered_usd.iloc[0]
            latest = float(selected_row['Close'])

            prev_df = usd[usd['Date'] < selected_datetime]

            if not prev_df.empty:
                prev = float(prev_df.iloc[-1]['Close'])
            else:
                prev = latest


        # INVALID
        else:
            st.error("Data Not Found for selected date")
            st.stop()

        change = latest - prev

        # METRICS
        col1, col2, col3, col4 = st.columns(4)

        arrow = "▲" if change > 0 else "▼"
        color = "normal" if change > 0 else "inverse"

        col1.metric("📅 Selected Day", f"₹ {latest:.2f}", f"{arrow} {change:.2f}", delta_color=color)
        col2.metric("🗓️ Yesterday", f"₹ {prev:.2f}")
        col3.metric("📈 Highest", f"₹ {float(usd['Close'].max()):.2f}")
        col4.metric("📉 Lowest", f"₹ {float(usd['Close'].min()):.2f}")

        #  NEXT DAY PREDICTION
        if model is not None:
            try:
                lag1 = latest
                lag2 = prev
                temp_prices = usd['Close'].tolist()
                lag3 = temp_prices[-3]

                ma7 = usd['Close'].rolling(7).mean().iloc[-1] if len(usd) >= 7 else lag1
                ma30 = usd['Close'].rolling(30).mean().iloc[-1] if len(usd) >= 30 else lag1
                day_of_week = selected_datetime.dayofweek
                input_data = create_input(lag1, lag2, lag3, ma7, ma30, day_of_week)
                prediction = predict_usd_next(latest, prev, usd)

                pred_change = prediction - latest

                arrow = "▲" if pred_change > 0 else "▼"
                color = "normal" if pred_change > 0 else "inverse"

                st.metric(
                    "🎯 Predicted Next Day",
                    f"₹ {prediction:.2f}",
                    f"{arrow} {abs(pred_change):.2f}",
                    delta_color=color
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # LAST 5 DAYS TABLE
        styled_subheader("Last 5 Days USD-INR")

        last5 = usd[usd['Date'] <= pd.to_datetime(selected_date)].tail(5).copy().reset_index(drop=True)

        rows = []

        for i in range(len(last5)):
            today_val = float(last5.loc[i, 'Close'])

            if i == 0:
                change_html = "-"
            else:
                prev_val = float(last5.loc[i-1, 'Close'])
                diff = today_val - prev_val

                if diff > 0:
                    change_html = f"<span style='color:#02ff99'>▲ ₹{abs(diff):.2f}</span>"
                else:
                    change_html = f"<span style='color:#ff4d4d'>▼ ₹{abs(diff):.2f}</span>"

            rows.append({
                "Date": last5.loc[i, 'Date'].date(),
                "Price": f"₹{today_val:.2f}",
                "Change": change_html
            })

        table_df = pd.DataFrame(rows)
        st.markdown(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        #  7 DAYS PREDICTION
        if model is not None:
            #  7 DAYS PREDICTION (FINAL FIXED)
            future_preds = []
            future_dates = []
            last_val = latest
            prev_val = prev

            for i in range(7):
                next_pred = predict_usd_next(last_val, prev_val, usd)
                # smoothing (VERY IMPORTANT)
                future_preds.append(round(next_pred, 2))
                future_dates.append(selected_datetime + timedelta(days=i+1))
                prev_val = last_val
                last_val = next_pred
            styled_subheader("7 Day USD Prediction")
            pred_rows = []
            for i in range(len(future_preds)):
                if i == 0:
                    prev_val = latest
                else:
                    prev_val = future_preds[i-1]

                curr = future_preds[i]
                diff = curr - prev_val

                if diff > 0:
                    change_html = f"<span style='color:#02ff99'>▲ ₹{abs(diff):.2f}</span>"
                else:
                    change_html = f"<span style='color:#ff4d4d'>▼ ₹{abs(diff):.2f}</span>"

                pred_rows.append({
                    "Date": future_dates[i].date(),
                    "Predicted Price": f"₹{curr:.2f}",
                    "Change": change_html
                })

            pred_df = pd.DataFrame(pred_rows)
            st.markdown(pred_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # GRAPH
        styled_subheader("USD-INR Trend")

        options = ["1W", "1M", "3M", "6M", "1Y", "ALL"]
        key_name = "usd_range"

        if key_name not in st.session_state:
            st.session_state[key_name] = "3M"

        left, c1, c2, c3, c4, c5, c6, right = st.columns([2,1,1,1,1,1,1,2])
        cols = [c1, c2, c3, c4, c5, c6]

        for i, opt in enumerate(options):
            if cols[i].button(opt, key=f"usd_{opt}"):
                st.session_state[key_name] = opt

        selected = st.session_state[key_name]

        if selected == "1W":
            usd_filtered = usd[usd['Date'] <= selected_datetime].tail(7)
        elif selected == "1M":
            usd_filtered = usd[usd['Date'] <= selected_datetime].tail(30)
        elif selected == "3M":
            usd_filtered = usd[usd['Date'] <= selected_datetime].tail(90)
        elif selected == "6M":
            usd_filtered = usd[usd['Date'] <= selected_datetime].tail(180)
        elif selected == "1Y":
            usd_filtered = usd[usd['Date'] <= selected_datetime].tail(365)
        else:
            usd_filtered = usd[usd['Date'] <= selected_datetime]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=usd_filtered['Date'],
            y=usd_filtered['Close'],
            mode='lines',
            line=dict(color='#2ecc71', width=3),
            name="USD-INR"
        ))

        fig.add_trace(go.Scatter(
            x=usd_filtered['Date'],
            y=usd_filtered['Close'],
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(46,204,113,0.15)'
        ))

        #  Prediction Line
        if model is not None:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_preds,
                mode='lines+markers',
                line=dict(color='#f39c12', width=3, dash='dash'),
                name="Prediction"
            ))

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date",
            yaxis_title="Price (₹)"
        )

        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"Error: {e}")
        
with tab5:
    styled_subheader("📊 Model Performance Dashboard")

    col1, col2, col3 = st.columns(3)

    # GOLD
    with col1:
        st.markdown("### 🪙 Gold Model")
        st.metric("MAE", f"{gold_metrics['MAE']:.2f}")
        st.metric("RMSE", f"{gold_metrics['RMSE']:.2f}")
        st.metric("R² Score", f"{gold_metrics['R2']:.4f}")

    # SILVER
    with col2:
        st.markdown("### 🔘 Silver Model")
        st.metric("MAE", f"{silver_metrics['MAE']:.2f}")
        st.metric("RMSE", f"{silver_metrics['RMSE']:.2f}")
        st.metric("R² Score", f"{silver_metrics['R2']:.4f}")

    # USD
    with col3:
        st.markdown("### 💱 USD Model")
        st.metric("MAE", f"{usd_metrics['MAE']:.2f}")
        st.metric("RMSE", f"{usd_metrics['RMSE']:.2f}")
        st.metric("R² Score", f"{usd_metrics['R2']:.4f}")

    st.markdown("---")

    st.success("Models evaluated using cross-validation (TimeSeriesSplit)")
#  FOOTER
st.markdown("---")
st.caption("© 2026 • Developed by Daksh Vasani | Advanced Analytics • Machine Learning • Financial Insights")
