# get_signal_alpaca_embedded.py
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame

# ======= EMBEDDED CREDENTIALS (as requested) =======
API_KEY = "PKCZOQQ5ZW6QR26GUIPODCGIT7"
API_SECRET = "GYwKaDmr4D2Lr7ZxHVNHDju32uUKcuGP8G1tsLvMJQXa"
BASE_URL = "https://paper-api.alpaca.markets/v2"
# ====================================================

TICKER = "AAPL"
WANT_BUSINESS_DAYS = 10

def _iso_utc(dt: datetime) -> str:
    """Return RFC3339-like UTC timestamp without microseconds, e.g. 2025-10-20T04:10:04Z"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace('+00:00', 'Z')

def make_client():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API credentials are not set in the script")
    return REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

def download_business_days(client, symbol, want=WANT_BUSINESS_DAYS, max_lookback_days=365):
    lookback_days = 15
    while lookback_days <= max_lookback_days:
        end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)
        start_s = _iso_utc(start_dt)
        end_s = _iso_utc(end_dt)
        bars = client.get_bars(symbol, TimeFrame.Day, start=start_s, end=end_s, limit=1000, feed='iex').df
        if bars is None or bars.empty:
            lookback_days += 7
            time.sleep(0.25)
            continue

        if isinstance(bars.index, pd.MultiIndex):
            if symbol in bars.index.get_level_values(0):
                df = bars.xs(symbol, level=0, drop_level=True)
            else:
                df = pd.DataFrame()
        else:
            df = bars.copy()

        if df.empty:
            lookback_days += 7
            time.sleep(0.25)
            continue

        # Make timezone-naive index and filter weekends
        df.index = pd.to_datetime(df.index).tz_convert(None)
        weekday_df = df[df.index.weekday < 5].sort_index()

        if len(weekday_df) >= want:
            return weekday_df.iloc[-want:]
        lookback_days += 7
        time.sleep(0.25)

    raise RuntimeError(f"Could not fetch {want} business days within {max_lookback_days} lookback days")

def close_series_from_bars(df):
    if "close" in df.columns:
        return df["close"].astype(float)
    if "Close" in df.columns:
        return df["Close"].astype(float)
    numeric_cols = df.select_dtypes("number").columns
    if len(numeric_cols) == 0:
        raise KeyError("No numeric columns found in bars to use as close")
    return df[numeric_cols[-1]].astype(float)

def compute_signal(symbol=TICKER):
    client = make_client()
    df = download_business_days(client, symbol, want=WANT_BUSINESS_DAYS)
    closes = close_series_from_bars(df).dropna()
    if len(closes) < 5:
        return {"ticker": symbol, "rows": len(closes), "signal": "HOLD", "reason": "not_enough_data"}
    last_close = float(closes.iloc[-1])
    avg5 = float(closes.iloc[-5:].mean())
    signal = "BUY" if last_close > avg5 else "SELL"
    return {
        "ticker": symbol,
        "rows": len(closes),
        "as_of": closes.index[-1].strftime("%Y-%m-%d"),
        "last_close": last_close,
        "avg5": avg5,
        "signal": signal,
    }

if __name__ == "__main__":
    result = compute_signal("AAPL")
    print("Ticker             :", result["ticker"])
    print("Rows fetched       :", result["rows"])
    print("As of              :", result.get("as_of"))
    print("Last trading close :", result.get("last_close"))
    print("5-day average      :", result.get("avg5"))
    print("Signal             :", result["signal"])
