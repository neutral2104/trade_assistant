import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def make_features(df):
    df = df.copy()

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_200"] = df["close"].rolling(200).mean()

    df["vol_20"] = df["return"].rolling(20).std()

    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]) / df["close"]

    # Target: regime
    df["regime"] = (df["close"] > df["ma_200"]).astype(int)

    df = df.dropna()
    return df

@st.cache_resource
def train_model(df):
    df = make_features(df)

    features = ["return","log_return","ma_20","ma_50","vol_20","range","body"]
    X = df[features]
    y = df["regime"]

    split = int(len(df) * 0.7)
    X_train, y_train = X.iloc[:split], y.iloc[:split]

    base = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base)
    model.fit(X_train, y_train)

    return model, features, df

def predict_today(df, model, features):
    latest = df.iloc[-1:][features]

    prob = model.predict_proba(latest)[0]
    pred = model.predict(latest)[0]

    if pred == 1:
        return "BULL", prob[1]
    else:
        return "BEAR", prob[0]


st.set_page_config(page_title="Regime Trading AI", layout="wide")

st.title("📈 Multi-Asset AI Regime Detector")

ASSETS = {
    "Gold (GLD)": "GLD",
    "Bitcoin (BTC)": "BTC-USD",
    "S&P 500 (SPY)": "SPY",
    "Nasdaq (QQQ)": "QQQ"
}

asset_name = st.sidebar.selectbox("Choose Market", list(ASSETS.keys()))
ticker = ASSETS[asset_name]

start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2010-01-01"))

@st.cache_data
def load_data(ticker, start):
    df = yf.download(ticker, start=start)
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns = df.columns.str.lower()
    df = df[['open','high','low','close','volume']]
    return df

df = load_data(ticker, start_date)

st.subheader(f"Data for {asset_name}")
st.write(df.tail())
st.line_chart(df["close"])

model, features, df_feat = train_model(df)
label, conf = predict_today(df_feat, model, features)


st.markdown("## 📊 Current Market Regime")

if label == "BULL":
    st.success("🟢 BULL MARKET")
else:
    st.error("🔴 BEAR MARKET")

st.metric("📅 Date", df_feat.index[-1].strftime("%Y-%m-%d"))

    
st.metric("📈 Confidence", f"{round(conf*100, 2)} %")

st.markdown("## 📉 Price Chart with Regime Overlay")

plot_df = df_feat.copy()

bull = plot_df["regime"] == 1
bear = plot_df["regime"] == 0

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14,6))

# Price line
ax.plot(plot_df.index, plot_df["close"], label="Price", color="white", alpha=0.8)

# MA200
ax.plot(plot_df.index, plot_df["ma_200"], label="MA200", color="orange", linestyle="--", alpha=0.8)

# Regime dots
ax.scatter(plot_df.index[bull], plot_df["close"][bull], color="green", s=8, label="Bull", alpha=0.7)
ax.scatter(plot_df.index[bear], plot_df["close"][bear], color="red", s=8, label="Bear", alpha=0.7)

ax.set_title(f"{asset_name} — Market Regime Overlay")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)

