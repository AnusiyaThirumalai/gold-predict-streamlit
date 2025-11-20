import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
import numpy as np

st.set_page_config(page_title="Gold Price Predictor", layout="centered")

# ---------- GOLD THEME UI ----------
st.markdown("""
    <h2 style='text-align:center; color:#FFD700;'>💰 Gold Price Prediction Dashboard</h2>
    <p style='text-align:center; font-size:18px;'>AI-powered Gold Prediction (India)</p>
    <hr style='border:1px solid #FFD700;'>
""", unsafe_allow_html=True)


# ---------- API URLs ----------
FLASK_API = "http://127.0.0.1:5000/predict"
USD_API = "https://api.exchangerate-api.com/v4/latest/USD"    # Free USD → INR API


# ========= (2) REAL-TIME FETCHING =========

st.subheader("🌍 Real-Time Data Fetch")
if st.button("Fetch Live USD/INR"):
    try:
        live_data = requests.get(USD_API).json()
        live_rate = live_data["rates"]["INR"]
        st.success(f"Live USD/INR: {live_rate}")
        usd_inr_default = live_rate
    except:
        st.error("Real-time API not working now!")
        usd_inr_default = 83.10
else:
    usd_inr_default = 83.10


# ========= USER INPUTS =========
st.subheader("📥 Enter Data for Prediction")

col1, col2 = st.columns(2)

with col1:
    usd_inr = st.number_input("Current USD/INR", value=usd_inr_default)
    usd_lag1 = st.number_input("Previous USD/INR", value=83.05)
    gold_lag1 = st.number_input("Gold Last Week (₹)", value=5050)

with col2:
    gold_lag2 = st.number_input("Gold 2 Weeks Ago (₹)", value=5025)
    gold_roll3 = st.number_input("3 Week Average (₹)", value=5030)
    gold_roll5 = st.number_input("5 Week Average (₹)", value=5015)



# ========= PREDICT BUTTON =========
st.subheader("📊 Prediction")
if st.button("Predict Gold Price"):
    with st.spinner("Predicting using ML model..."):
        
        payload = {
            "USD_INR": usd_inr,
            "USD_lag1": usd_lag1,
            "Gold_lag1": gold_lag1,
            "Gold_lag2": gold_lag2,
            "Gold_roll3": gold_roll3,
            "Gold_roll5": gold_roll5
        }

        res = requests.post(FLASK_API, json=payload).json()

        lr_pred = res["LinearRegression_Prediction"]
        rf_pred = res["RandomForest_Prediction"]

        st.success("Prediction successful!")

        # ---- display results ----
        col3, col4 = st.columns(2)
        with col3:
            st.metric(label="Linear Regression Prediction (₹)", value=f"{lr_pred:,.2f}")
        with col4:
            st.metric(label="Random Forest Prediction (₹)", value=f"{rf_pred:,.2f}")


        # ========= (3) COMPARISON CHART =========
        df_chart = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "Predicted Price": [lr_pred, rf_pred]
        })

        fig = px.bar(
            df_chart,
            x="Model",
            y="Predicted Price",
            color="Model",
            title="Gold Price Prediction Comparison",
            color_discrete_sequence=["#FFD700", "#FFA500"]
        )
        st.plotly_chart(fig)


        # ========= (4) FUTURE FORECAST (Next 7 days) =========
        st.subheader("🔮 7-Day Forecast (Simple Projection)")
        
        # Simple trend-based forecast
        daily_change = (gold_lag1 - gold_lag2) / 7  
        future_dates = []
        future_prices = []

        last_price = lr_pred

        for i in range(1, 8):
            next_date = datetime.date.today() + datetime.timedelta(days=i)
            next_price = last_price + daily_change
            future_dates.append(next_date)
            future_prices.append(next_price)
            last_price = next_price

        df_forecast = pd.DataFrame({
            "Date": future_dates,
            "Predicted Gold Price (₹)": future_prices
        })

        fig_forecast = px.line(
            df_forecast,
            x="Date",
            y="Predicted Gold Price (₹)",
            title="7-Day Future Gold Price Forecast",
            markers=True
        )
        st.plotly_chart(fig_forecast)

        st.write(df_forecast)
