import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- PAGE ----------
st.set_page_config(page_title="Stock Predictor")

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}
h1 {
    text-align: center;
    color: white;
}
.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
    color: white;
    border-radius: 8px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>📈 Stock Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------- STOCK LIST ----------
stocks = [
    "AAPL","TSLA","GOOGL","MSFT","AMZN",
    "META","NVDA","NFLX","AMD",
    "TCS.NS","RELIANCE.NS","INFY.NS","HDFCBANK.NS"
]

# ---------- INPUT ----------
col1, col2, col3 = st.columns(3)

with col1:
    stock = st.selectbox("Stock", stocks)

with col2:
    days = st.number_input("Days", min_value=1, max_value=30, value=7)

with col3:
    st.write("")
    predict_btn = st.button("Predict")

# ---------- DATA ----------
df = yf.download(stock, start="2020-01-01")

if df.empty:
    st.error("Data not loaded")
    st.stop()

# ---------- GRAPH ----------
st.subheader(f"{stock} Price History")
st.line_chart(df['Close'])

# ---------- ML ----------
data = df[['Close']].copy()

# shift based on days
data['Prediction'] = data['Close'].shift(-int(days))

# remove NaN
data.dropna(inplace=True)

# safety check
if len(data) < 20:
    st.warning("Not enough data")
    st.stop()

X = data[['Close']].values.reshape(-1, 1)
y = data['Prediction'].values

model = LinearRegression()
model.fit(X, y)

# ---------- PREDICTION ----------

if predict_btn:
    try:
        last_price = float(df['Close'].iloc[-1])

        future_prices = []
        current_price = last_price

        for i in range(int(days)):
            pred = model.predict(np.array([[current_price]]))[0]
            future_prices.append(float(pred))
            current_price = pred

        # ✅ FIXED LINE
        history = list(df['Close'])
        full_data = history + future_prices

        st.subheader("Prediction Graph")
        st.line_chart(full_data)

        st.success(f"📊 Predicted Price after {days} days: {future_prices[-1]:.2f}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
