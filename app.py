import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Price Predictor")

# ---------- SIDEBAR ----------
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

# ---------- STOCK LIST ----------
stocks = [
    "AAPL","TSLA","GOOGL","MSFT","AMZN",
    "META","NVDA","NFLX","AMD",
    "TCS.NS","RELIANCE.NS","INFY.NS","HDFCBANK.NS"
]

col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("Select Stock", stocks)

with col2:
    days = st.slider("Days to Predict", 1, 30, 7)

# ---------- SESSION STATE FOR BUTTON ----------
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button("🔮 Predict Future Price"):
    st.session_state.predict_clicked = True

# ---------- MAIN LOOP ----------
placeholder = st.empty()

while True:
    with placeholder.container():

        # ---------- LOAD LIVE DATA ----------
        df = yf.download(stock, period="1d", interval="1m")

        if df is None or df.empty:
            st.warning("Waiting for live data...")
            time.sleep(2)
            st.rerun()

        # ---------- CLEAN CLOSE DATA ----------
        close_data = df['Close']

        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]

        close_data = close_data.dropna()

        if len(close_data) < 20:
            st.warning("Not enough data yet...")
            time.sleep(2)
            st.rerun()

        # ---------- CURRENT PRICE ----------
        current_price = float(close_data.values[-1])
        st.metric("Current Price", f"{current_price:.2f}")
        st.caption("🔄 Auto-refreshing live data")

        # ---------- LIVE GRAPH ----------
        st.subheader("📊 Live Price Chart")
        st.line_chart(close_data)

        # ---------- FEATURE ENGINEERING ----------
        window = 10
        X = []
        y = []

        for i in range(window, len(close_data)):
            X.append(close_data.iloc[i-window:i].values)
            y.append(close_data.iloc[i])

        X = np.array(X)
        y = np.array(y)

        # ---------- MODEL ----------
        model = LinearRegression()
        model.fit(X, y)

        # ---------- PREDICTION ----------
        if st.session_state.predict_clicked:

            last_window = close_data.iloc[-window:].values
            future = []

            for i in range(days):
                pred = model.predict([last_window])[0]
                future.append(pred)
                last_window = np.append(last_window[1:], pred)

            st.subheader("📉 Prediction Graph")

            history = list(close_data.tail(100))
            full = history + future

            st.line_chart(full)

            st.success(f"📊 Predicted Price after {days} days: {future[-1]:.2f}")

    # ---------- AUTO REFRESH ----------
    time.sleep(refresh_interval)
    st.rerun()
