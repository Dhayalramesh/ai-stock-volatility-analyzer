import streamlit as st
import yfinance as yf
import pandas as pd
from ml_predict import predict_stock

# 🔹 PAGE CONFIG
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# 🔹 HEADER
st.title("🚀 AI Stock Volatility Analyzer")
st.markdown("### 🧠 Real-time AI-powered financial risk engine")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# 🔹 INPUT SECTION
col1, col2 = st.columns([3, 1])

with col1:
    stock = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS)")

with col2:
    analyze_btn = st.button("Analyze Stock")

# 🔹 ANALYSIS
if analyze_btn:

    if not stock:
        st.warning("Please enter a stock symbol")
    else:
        stock = stock.upper()

        # 🔥 LOADING
        with st.spinner("🧠 AI analyzing market data..."):
            df = yf.download(stock, period="6mo")

        if df.empty:
            st.error("❌ Invalid stock or no data available")
        else:
            # 🔥 FIX MULTIINDEX
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 🔥 CLEAN DATA
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df.dropna(inplace=True)

            prediction = predict_stock(df)

            if prediction is None:
                st.warning("⚠️ Not enough data to predict")
            else:
                st.markdown("## 📊 AI Prediction Dashboard")
                st.markdown("<br>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # 🔹 LEFT PANEL
                with col1:
                    st.markdown("### 📌 Prediction Result")

                    # BIG NUMBER
                    st.markdown(f"# 📊 {prediction:.4f}")
                    st.caption("Predicted future volatility (log scale)")

                    # RISK LEVEL
                    if prediction < -4.7:
                        st.success("🟢 LOW RISK (Stable Stock)")
                    elif prediction < -4.4:
                        st.warning("🟡 MEDIUM RISK")
                    else:
                        st.error("🔴 HIGH RISK (Volatile Stock)")

                    # VISUAL BAR
                    risk_score = min(max((prediction + 5) / 1, 0), 1)
                    st.progress(risk_score)

                    # CONFIDENCE LABEL
                    st.info("Model Confidence: Medium (based on historical patterns)")

                    st.markdown("---")

                    st.markdown("### 📖 Interpretation")
                    st.info(
                        "Lower value → Less volatility (safe)\n\n"
                        "Higher value → More volatility (risky)"
                    )

                # 🔹 RIGHT PANEL (CHART FIXED)
                with col2:
                    st.markdown("### 📈 Price Trend (6 Months)")

                    if "Close" in df.columns and not df["Close"].empty:
                        st.line_chart(df["Close"])
                    else:
                        st.warning("Chart data unavailable")

# 🔹 DIVIDER
st.markdown("---")

# 🔥 QUICK MARKET SCAN
st.subheader("🔥 Quick Market Scan")
st.markdown("Analyze top stocks instantly using AI")

scan_btn = st.button("Scan Top Stocks")

if scan_btn:

    stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    results = []

    with st.spinner("Scanning market using AI..."):
        for s in stocks:
            df = yf.download(s, period="6mo")

            if not df.empty:

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                df.dropna(inplace=True)

                pred = predict_stock(df)

                if pred is not None:
                    results.append((s, pred))

    if results:
        results = sorted(results, key=lambda x: x[1])

        st.markdown("## 📊 Market Ranking")

        for s, p in results:
            if p < -4.7:
                st.markdown(f"### 🟢 {s} → {p:.4f} (LOW RISK)")
            elif p < -4.4:
                st.markdown(f"### 🟡 {s} → {p:.4f} (MEDIUM RISK)")
            else:
                st.markdown(f"### 🔴 {s} → {p:.4f} (HIGH RISK)")
    else:
        st.error("No valid stocks found")

# 🔹 FOOTER
st.markdown("---")
st.markdown(
    "⚡ Built by Dhayal SR | IIIT Dharwad | AI + Finance Project"
)