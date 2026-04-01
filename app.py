import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- PAGE ----------
st.set_page_config(page_title="Stock Predictor")

# ---------- TITLE ----------
st.title("📈 Stock Price Predictor")

# ---------- STOCK LIST ----------
stocks = [
    "AAPL","TSLA","GOOGL","MSFT","AMZN",
    "META","NVDA","NFLX","AMD",
    "TCS.NS","RELIANCE.NS","INFY.NS","HDFCBANK.NS"
]

# ---------- INPUT ----------
stock = st.selectbox("Select Stock", stocks)
days = st.number_input("Days to Predict", min_value=1, max_value=30, value=7)
predict_btn = st.button("Predict")

# ---------- DATA LOADING (FIXED) ----------
@st.cache_data
def load_data(stock):
    try:
        df = yf.Ticker(stock).history(period="5y")   # more reliable
        return df
    except:
        return None

df = load_data(stock)

if df is None or df.empty:
    st.error("❌ Data not loaded. Check internet or try another stock.")
    st.stop()

# Ensure numeric
df['Close'] = df['Close'].astype(float)

# ---------- GRAPH ----------
st.subheader(f"{stock} Price History")
st.line_chart(df['Close'])

# ---------- ML ----------
data = df[['Close']].copy()

# Create future prediction column
data['Prediction'] = data['Close'].shift(-int(days))

# Remove NaN
data.dropna(inplace=True)

if len(data) < 20:
    st.warning("Not enough data")
    st.stop()

X = data[['Close']].values
y = data['Prediction'].values

# Train model
model = LinearRegression()
model.fit(X, y)

# ---------- PREDICTION ----------
if predict_btn:
    try:
        last_price = float(df['Close'].iloc[-1])

        future_prices = []
        current_price = last_price

        for _ in range(int(days)):
            pred = model.predict(np.array([[current_price]]))
            pred_value = float(pred[0])   # FIXED

            future_prices.append(pred_value)
            current_price = pred_value

        # Combine history + prediction
        history = df['Close'].tolist()
        full_data = history + future_prices

        st.subheader("Prediction Graph")
        st.line_chart(full_data)

        st.success(f"📊 Predicted Price after {days} days: {future_prices[-1]:.2f}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
