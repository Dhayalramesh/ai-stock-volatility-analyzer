import pandas as pd
import os

def predict_volatility(stock):
    stock = stock.upper()
    file_path = f"data/{stock}_with_vix.csv"

    if not os.path.exists(file_path):
        print("❌ Data file not found. Run collect_data.py first.")
        return

    # 🔹 FIX: handle weird CSV format
    df = pd.read_csv(file_path)

    # Try to fix multi-level column issue
    df.columns = [col if isinstance(col, str) else col[1] for col in df.columns]

    # 🔹 Convert Close to numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Drop invalid rows
    df.dropna(subset=["Close"], inplace=True)

    # Calculate returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

    latest_vol = df["Volatility_10"].iloc[-1]

    if pd.isna(latest_vol):
        print("❌ Not enough data.")
        return

    # Risk classification
    if latest_vol > 0.02:
        risk = "HIGH"
    elif latest_vol > 0.01:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    print("\n📊 STOCK ANALYSIS RESULT")
    print(f"Stock: {stock}")
    print(f"Predicted Volatility: {latest_vol:.4f}")
    print(f"Risk Level: {risk}")

if __name__ == "__main__":
    stock = input("Enter stock ticker: ")
    predict_volatility(stock)