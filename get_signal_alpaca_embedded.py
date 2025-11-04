# streamlit_app.py
"""
Free AI Trader — Full merged file
- Floating Help widget inside overlay iframe (height=400)
- Improved SELL logic using negative historical returns where available
- Highlighted entry/exit prices, suggested quantity, exit day, estimated profit %
- Educational / paper-trading only. Test on Alpaca PAPER with Dry run before live trading.
"""
import os
import time
import pickle
import io
from datetime import datetime, timedelta, timezone

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from alpaca_trade_api.rest import REST, TimeFrame

# ======= Embedded Alpaca Credentials (paper API) =======
ALPACA_API_KEY = "PKFOK2IXE2XR7QWISVFDUAPDBK"
ALPACA_API_SECRET = "JCXLrYaarzYDkLUG1ZJakgGySCuFni5tJgWSC37PXdUA"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
# =======================================================

st.set_page_config(page_title="Free AI Trader — Timing & Expected Profit", layout="wide")

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "logreg.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
LOOKBACK_MAP = {"6m": 183, "3m": 92, "1m": 31, "2w": 14}
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# ---------------- Helpers ----------------
def _iso_utc(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def make_client():
    return REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL, api_version="v2")

def download_business_days(client, symbol, want_days=60, max_lookback_days=365):
    lookback_days = max(want_days + 7, 30)
    while lookback_days <= max_lookback_days:
        end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)
        start_s = _iso_utc(start_dt); end_s = _iso_utc(end_dt)
        bars = client.get_bars(symbol, TimeFrame.Day, start=start_s, end=end_s, limit=1000, feed="iex").df
        if bars is None or bars.empty:
            lookback_days += 7; time.sleep(0.25); continue
        if isinstance(bars.index, pd.MultiIndex):
            if symbol in bars.index.get_level_values(0):
                df = bars.xs(symbol, level=0, drop_level=True)
            else:
                df = pd.DataFrame()
        else:
            df = bars.copy()
        if df.empty:
            lookback_days += 7; time.sleep(0.25); continue
        df.index = pd.to_datetime(df.index).tz_convert(None)
        weekday_df = df[df.index.weekday < 5].sort_index()
        if len(weekday_df) >= want_days:
            return weekday_df.iloc[-want_days:]
        lookback_days += 7; time.sleep(0.25)
    raise RuntimeError("Could not fetch required business days")

def close_series_from_bars(df):
    if "close" in df.columns:
        return df["close"].astype(float)
    if "Close" in df.columns:
        return df["Close"].astype(float)
    numeric_cols = df.select_dtypes("number").columns
    if len(numeric_cols) == 0:
        raise KeyError("No numeric columns for close")
    return df[numeric_cols[-1]].astype(float)

# ---------------- Feature / Label / Model utils ----------------
def feature_engineer(closes):
    df = pd.DataFrame({"close": closes})
    df["ret1"] = df["close"].pct_change(1)
    df["ret2"] = df["close"].pct_change(2)
    df["ret3"] = df["close"].pct_change(3)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["vol5"] = df["ret1"].rolling(5).std()
    df["mom"] = df["close"] / df["close"].rolling(5).mean() - 1
    return df.dropna()

def make_labels(closes, horizon_days=1, threshold=0.0):
    future = closes.shift(-horizon_days)
    fut_ret = future / closes - 1.0
    labels = (fut_ret > threshold).astype(int)
    return labels[:-horizon_days], fut_ret[:-horizon_days]

def align_features_labels(feats, labels):
    idx = feats.index.intersection(labels.index)
    return feats.loc[idx].copy(), labels.loc[idx].copy()

def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    split_at = max(1, int(len(Xs) * 0.8))
    model = LogisticRegression(max_iter=1000)
    model.fit(Xs[:split_at], y[:split_at])
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f: pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f: pickle.dump(scaler, f)
    ypred = model.predict(Xs[split_at:]) if len(Xs[split_at:]) > 0 else np.array([])
    val_acc = float(accuracy_score(y[split_at:], ypred)) if len(ypred) > 0 else None
    return model, scaler, {"val_accuracy": val_acc}

def load_model():
    with open(MODEL_FILE, "rb") as f: model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)
    return model, scaler

def predict_prob(model, scaler, feats):
    Xs = scaler.transform(feats.values)
    probs = model.predict_proba(Xs)[:, 1]
    return float(probs[-1])

# ---------------- Backtester ----------------
def simulate_trades(closes, signals, initial_capital=10000.0, commission_per_share=0.0, slippage_pct=0.0, pos_size_frac=0.1):
    df = pd.DataFrame({"close": closes}).copy()
    df = df.loc[signals.index]
    df["signal"] = signals
    df = df.dropna()
    cash = initial_capital; shares = 0
    equity_records = []; trades = []
    for idx, row in df.iterrows():
        price = float(row["close"])
        desired_shares = int((initial_capital * pos_size_frac * row["signal"]) // price) if price > 0 else 0
        delta = desired_shares - shares
        if delta != 0:
            slippage = price * slippage_pct
            exec_price = price + slippage if delta > 0 else price - slippage
            commission = abs(delta) * commission_per_share
            cost = delta * exec_price
            cash -= (cost + commission)
            trades.append({
                "date": idx, "side": "buy" if delta > 0 else "sell",
                "delta_shares": int(delta), "exec_price": float(exec_price),
                "commission": float(commission), "cash_after": float(cash)
            })
            shares = desired_shares
        equity = cash + shares * price
        equity_records.append({"date": idx, "equity": float(equity), "cash": float(cash), "shares": int(shares)})
    equity_df = pd.DataFrame(equity_records).set_index("date") if equity_records else pd.DataFrame(columns=["equity"])
    trades_df = pd.DataFrame(trades)
    if not equity_df.empty:
        equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)
        total_return = (equity_df["equity"].iloc[-1] / initial_capital) - 1.0
        cummax = equity_df["equity"].cummax()
        drawdown = cummax - equity_df["equity"]
        max_drawdown_abs = float(drawdown.max())
        max_drawdown_pct = float((drawdown.max() / cummax.max()) if cummax.max() > 0 else 0.0)
        sharpe = (equity_df["returns"].mean() / (equity_df["returns"].std() + 1e-12)) * np.sqrt(252) if equity_df["returns"].std() > 0 else 0.0
    else:
        total_return = 0.0; max_drawdown_abs = 0.0; max_drawdown_pct = 0.0; sharpe = 0.0
    wins, losses = [], []; buy_stack = []
    for _, t in trades_df.iterrows():
        if t["side"] == "buy":
            buy_stack.append(t)
        else:
            if buy_stack:
                b = buy_stack.pop(0)
                qty = min(abs(b["delta_shares"]), abs(t["delta_shares"]))
                pnl = (t["exec_price"] - b["exec_price"]) * qty - (b["commission"] + t["commission"])
                if pnl >= 0: wins.append(pnl)
                else: losses.append(pnl)
    win_rate = (len(wins) / (len(wins) + len(losses))) if (len(wins) + len(losses)) > 0 else None
    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None
    metrics = {
        "total_return": float(total_return),
        "max_drawdown_abs": float(max_drawdown_abs),
        "max_drawdown_pct": float(max_drawdown_pct),
        "sharpe": float(sharpe),
        "num_trades": int(len(trades_df)),
        "win_rate": float(win_rate) if win_rate is not None else None,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }
    return trades_df, equity_df, metrics

# ---------------- Help iframe HTML (overlay content) ----------------
HELP_IFRAME_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root { font-family: Arial, Helvetica, sans-serif; }
  #help-root { position: fixed; right: 18px; bottom: 18px; z-index: 9999999; }
  #help-button {
    width:64px; height:36px; border-radius:20px; border:none;
    background: linear-gradient(90deg,#0d6efd,#2b8cff); color:#fff; font-weight:600; cursor:pointer;
    box-shadow: 0 6px 18px rgba(13,110,253,0.25);
  }
  #help-panel {
    width:320px; height:264px; /* ~7cm */
    position: fixed; right: 18px; bottom: 66px; z-index: 9999999;
    background: #ffffff; border-radius: 8px; box-shadow: 0 8px 30px rgba(0,0,0,0.16);
    overflow: hidden; display:none; font-size:13px;
  }
  #help-header { background:#0d6efd; color:white; padding:8px 10px; font-weight:600; }
  #help-body { padding:8px; height: calc(100% - 96px); overflow:auto; background:#fafafa; }
  #help-input-row { display:flex; padding:8px; gap:6px; background:#fff; }
  #help-input { flex:1; padding:6px 8px; border:1px solid #ddd; border-radius:6px; }
  .help-msg-user { text-align:right; background:#e9f5ff; padding:6px 8px; border-radius:8px; margin:6px 0; }
  .help-msg-bot { text-align:left; background:#f1f3f5; padding:6px 8px; border-radius:8px; margin:6px 0; }
  .help-footer { padding:6px 8px; font-size:12px; color:#666; }
</style>
</head>
<body>
<div id="help-root" aria-hidden="false">
  <button id="help-button" title="Help">Help</button>
  <div id="help-panel" role="dialog" aria-label="Help panel">
    <div id="help-header">Help</div>
    <div id="help-body">
      <div id="help-history">
        <div class="help-msg-bot">Hi — I can explain controls and outputs on this page. Try "Lookback", "Label threshold", "Why no trades?"</div>
      </div>
    </div>
    <div id="help-input-row">
      <input id="help-input" placeholder="Ask: Lookback, Label threshold, Drawdown..." />
      <button id="help-send">Send</button>
    </div>
    <div class="help-footer">Local help — runs in your browser only</div>
  </div>
</div>

<script>
(function(){
  const button = document.getElementById("help-button");
  const panel = document.getElementById("help-panel");
  const send = document.getElementById("help-send");
  const input = document.getElementById("help-input");
  const history = document.getElementById("help-history");
  const HELP_RESPONSES = {
    "tickers": "Tickers: up to 5 US-listed symbols separated by commas.",
    "lookback": "Lookback: historical window used to build features.",
    "label threshold": "Label threshold: minimum future return to label positive.",
    "no trades": "No trades often means model produced no positive signals. Try lowering threshold or increasing lookback.",
    "place orders": "Place Orders: submits MARKET orders to Alpaca PAPER account when confirmed. Use Dry run first."
  };
  function appendMessage(text, from){const d=document.createElement("div");d.className= from==="user"? "help-msg-user":"help-msg-bot";d.innerHTML=text.replace(/\n/g,"<br>");history.appendChild(d);history.scrollTop=history.scrollHeight;}
  function getAnswer(q){ if(!q) return "Say something like: 'Lookback' or 'Label threshold'."; const t=q.toLowerCase(); for(const k in HELP_RESPONSES) if(t.indexOf(k)!==-1) return HELP_RESPONSES[k]; return "No specific answer. Try 'lookback', 'label threshold', 'no trades'."; }
  button.addEventListener("click", ()=>{ panel.style.display = panel.style.display === "block" ? "none" : "block"; if(panel.style.display === "block") input.focus(); });
  send.addEventListener("click", ()=>{ const q=input.value.trim(); if(!q) return; appendMessage(q,"user"); const a=getAnswer(q); setTimeout(()=>appendMessage(a,"bot"),150); input.value=""; });
  input.addEventListener("keydown", (e)=>{ if(e.key==="Enter"){ send.click(); e.preventDefault(); }});
})();
</script>
</body>
</html>
"""

# ---------------- UI Controls ----------------
st.title("Free AI Trader — Timing & Expected Profit")
st.markdown("Educational app. Suggestions show when to buy today and when to sell (today/tomorrow/or later) plus estimated profit %.")

with st.sidebar:
    st.header("Selection and Model")
    tickers_txt = st.text_input("Tickers (comma-separated, up to 5)", value=",".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()][:5]
    lookback = st.selectbox("Lookback span", options=list(LOOKBACK_MAP.keys()), index=2)
    horizon = st.radio("Prediction horizon (days)", options=[1, 2, 3], index=0)
    percent_threshold = st.number_input("Label threshold (%)", min_value=0.0, value=0.0, step=0.1)
    threshold_decimal = float(percent_threshold) / 100.0
    retrain = st.checkbox("Retrain model", value=False)

    st.markdown("---")
    st.header("Simulator & Order Settings")
    commission_per_share = st.number_input("Commission per share (absolute)", min_value=0.0, value=0.0, step=0.01)
    slippage_pct = st.number_input("Slippage per trade (%)", min_value=0.0, value=0.0, step=0.01) / 100.0
    initial_capital = st.number_input("Initial capital", min_value=100.0, value=10000.0, step=100.0)
    pos_size_frac = st.number_input("Position size fraction (0-1)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    qty_mode = st.selectbox("Quantity mode", ["fixed_shares", "dollar_amount"])
    fixed_qty = st.number_input("Fixed shares (if selected)", min_value=1, value=1)
    dollar_amt = st.number_input("Dollar amount per ticker (if selected)", min_value=1.0, value=50.0)
    st.markdown("---")
    run_button = st.button("Run Predictions & Backtest")
    place_button = st.button("Place Orders (Paper)")

# Alpaca client
try:
    client = make_client()
except Exception as e:
    st.error(f"Alpaca client error: {e}")
    st.stop()

results = []
per_stock_data = {}

if run_button:
    st.info("Running predictions and backtests; this may take a few seconds per ticker.")
    for symbol in tickers:
        if not symbol: continue
        try:
            want_days = LOOKBACK_MAP.get(lookback, 31) + 40
            bars = download_business_days(client, symbol, want_days)
            closes = close_series_from_bars(bars).dropna()
            if len(closes) < 40:
                per_stock_data[symbol] = {"error":"Not enough raw data"}
                continue

            feats = feature_engineer(closes)
            labels, fut_ret = make_labels(closes, horizon_days=horizon, threshold=threshold_decimal)
            feats_aligned, labels_aligned = align_features_labels(feats, labels)

            fut_ret_aligned = fut_ret.loc[labels_aligned.index] if not fut_ret.empty else pd.Series(dtype=float)
            # compute positive/negative stats
            pos_mask = fut_ret_aligned > 0
            neg_mask = fut_ret_aligned < 0
            mean_pos_future_ret = float(fut_ret_aligned[pos_mask].mean()) if pos_mask.sum() > 0 else None
            mean_neg_future_ret = float(fut_ret_aligned[neg_mask].mean()) if neg_mask.sum() > 0 else None
            pos_count = int(pos_mask.sum())
            neg_count = int(neg_mask.sum())

            if len(feats_aligned) < 12:
                per_stock_data[symbol] = {"error": f"Insufficient aligned samples ({len(feats_aligned)})", "aligned_samples": len(feats_aligned)}
                continue

            X = feats_aligned.values; y = labels_aligned.values
            if retrain or not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
                model, scaler, model_metrics = train_model(X, y)
                trained = True
            else:
                try:
                    model, scaler = load_model(); model_metrics = None; trained = False
                except Exception:
                    model, scaler, model_metrics = train_model(X, y); trained = True

            prob = predict_prob(model, scaler, feats_aligned)
            signal = "BUY" if prob > 0.55 else ("SELL" if prob < 0.45 else "HOLD")

            Xs = scaler.transform(feats_aligned.values)
            preds = model.predict(Xs)
            signals = pd.Series(preds, index=feats_aligned.index).map({1: 1, 0: 0})

            trades_df, equity_df, metrics = simulate_trades(
                closes.loc[signals.index],
                signals,
                initial_capital=initial_capital,
                commission_per_share=commission_per_share,
                slippage_pct=slippage_pct,
                pos_size_frac=pos_size_frac
            )

            last_close = float(closes.iloc[-1])
            small_buffer = 0.001
            sell_fallback_buffer = 0.002

            # --- Improved recommendation logic (BUY uses positive mean, SELL uses negative mean) ---
            trust_pos = pos_count >= 5 and mean_pos_future_ret is not None
            trust_neg = neg_count >= 5 and mean_neg_future_ret is not None

            if signal == "BUY":
                if trust_pos:
                    expected_profit = mean_pos_future_ret
                    suggested_entry_price = last_close * (1.0 + small_buffer)
                    suggested_exit_price = last_close * (1.0 + expected_profit)
                    profit_note = "Estimated from historical positive-labeled returns"
                else:
                    expected_profit = small_buffer
                    suggested_entry_price = last_close * (1.0 + small_buffer)
                    suggested_exit_price = last_close * (1.0 + small_buffer)
                    profit_note = "Fallback conservative estimate"
                entry_day = "today"
                exit_day = "tomorrow" if horizon == 1 else f"in {horizon} trading days"

            elif signal == "SELL":
                if trust_neg:
                    # mean_neg_future_ret is negative (e.g., -0.02). For SELL expected_profit = -mean_neg_future_ret
                    expected_profit = -mean_neg_future_ret
                    suggested_entry_price = last_close * (1.0 - small_buffer)
                    suggested_exit_price = last_close * (1.0 - mean_neg_future_ret)
                    profit_note = "Estimated from historical negative-labeled returns"
                else:
                    expected_profit = sell_fallback_buffer
                    suggested_entry_price = last_close * (1.0 - small_buffer)
                    suggested_exit_price = last_close * (1.0 - sell_fallback_buffer)
                    profit_note = "Fallback conservative estimate"
                entry_day = "today"
                exit_day = "today"
            else:
                expected_profit = 0.0
                suggested_entry_price = None
                suggested_exit_price = None
                entry_day = "n/a"
                exit_day = "n/a"
                profit_note = "No action"

            expected_profit_pct = float(expected_profit) * 100.0
            suggested_entry_price = round(suggested_entry_price, 4) if suggested_entry_price is not None else None
            suggested_exit_price = round(suggested_exit_price, 4) if suggested_exit_price is not None else None

            if qty_mode == "fixed_shares":
                suggested_qty = int(fixed_qty)
            else:
                suggested_qty = max(1, int(dollar_amt // last_close)) if last_close > 0 else int(fixed_qty)

            metadata = {
                "aligned_samples": len(feats_aligned),
                "pos_count": pos_count,
                "neg_count": neg_count,
                "mean_pos_future_ret": round(mean_pos_future_ret,4) if mean_pos_future_ret is not None else None,
                "mean_neg_future_ret": round(mean_neg_future_ret,4) if mean_neg_future_ret is not None else None,
                "prob_up": round(prob,4),
                "signal": signal,
                "trained_now": trained,
                "last_close": last_close,
                "suggested_entry_price": suggested_entry_price,
                "suggested_exit_price": suggested_exit_price,
                "suggested_qty": suggested_qty,
                "entry_day": entry_day,
                "exit_day": exit_day,
                "expected_profit_pct": round(expected_profit_pct,3),
                "profit_note": profit_note
            }

            per_stock_data[symbol] = {"metadata": metadata, "trades": trades_df, "equity": equity_df, "metrics": metrics}
            results.append({"ticker": symbol, "signal": signal, "prob_up": round(prob,4), "aligned": len(feats_aligned)})

        except Exception as e:
            per_stock_data[symbol] = {"error": str(e)}

    st.success("Run complete")

# Quick results
if results:
    st.markdown("## Quick results")
    st.dataframe(pd.DataFrame(results))

# Per-stock display with timing and expected profit
if per_stock_data:
    st.markdown("## Per-stock recommendations (timing + expected profit)")
    for sym, info in per_stock_data.items():
        st.markdown(f"### {sym}")
        if "error" in info:
            st.error(info["error"])
            continue
        md = info["metadata"]
        summary = [
            f"Signal: **{md['signal']}** (prob up {md['prob_up']*100:.2f}%)",
            f"Aligned samples: {md['aligned_samples']}",
            f"Positive labels: {md.get('pos_count',0)}",
            f"Negative labels: {md.get('neg_count',0)}"
        ]
        if md.get("trained_now"):
            summary.append("Model: retrained this run")
        else:
            summary.append("Model: loaded from artifacts")
        st.markdown(" · ".join(summary))

        st.markdown("**Recommendation (timing & expected profit)**")
        if md["signal"] != "HOLD":
            entry_html = md["suggested_entry_price"] and f'<span style="display:inline-block;padding:4px 8px;border-radius:6px;background:#fff3cd;font-weight:700;color:#7a4a00;">Entry ≈ {md["suggested_entry_price"]}</span>' or ""
            exit_html = md["suggested_exit_price"] and f'<span style="display:inline-block;padding:4px 8px;border-radius:6px;background:#e6ffed;color:#0f6d2a;font-weight:700;margin-left:8px;">Exit ≈ {md["suggested_exit_price"]}</span>' or ""
            profit_color = "#0f6d2a" if md["expected_profit_pct"] >= 0 else "#a00"
            profit_html = f'<span style="display:inline-block;padding:4px 8px;border-radius:6px;background:{"#e6ffed" if md["expected_profit_pct"]>=0 else "#ffecec"};color:{profit_color};font-weight:700;margin-left:8px;">{md["expected_profit_pct"]}%</span>'
            st.markdown(f"- **Action:** **{md['signal']}**  - **Entry day:** **{md['entry_day']}**  {entry_html}", unsafe_allow_html=True)
            st.markdown(f"- **Suggested exit:** **{md['exit_day']}**  {exit_html}", unsafe_allow_html=True)
            st.markdown(f"- **Estimated profit:** {profit_html}  - _{md.get('profit_note','estimate')}_", unsafe_allow_html=True)
            st.markdown(f"- **Suggested quantity:** {md['suggested_qty']}")
            st.info("Suggestion only. Use Dry run to test order placement with Alpaca PAPER.")
        else:
            st.markdown("No action recommended today (HOLD).")

        metrics = info["metrics"]
        metrics_display = {
            "Total return (%)": f"{metrics['total_return']*100:.3f}%" if metrics.get("total_return") is not None else "N/A",
            "Max drawdown (abs)": f"{metrics['max_drawdown_abs']:.4f}",
            "Max drawdown (%)": f"{metrics['max_drawdown_pct']*100:.3f}%",
            "Sharpe (est)": f"{metrics['sharpe']:.4f}",
            "Simulated trades": f"{metrics['num_trades']}"
        }
        st.table(pd.DataFrame.from_dict(metrics_display, orient="index", columns=["Value"]))

        eq = info["equity"]
        if not eq.empty:
            st.line_chart(eq["equity"])
        else:
            st.info("No simulated trades in equity series.")

        trades = info["trades"]
        if not trades.empty:
            st.markdown("**Simulated trade log**")
            st.dataframe(trades)
            csv_buf = io.StringIO(); trades.to_csv(csv_buf, index=False)
            st.download_button(label=f"Download {sym} trade log CSV", data=csv_buf.getvalue().encode(), file_name=f"{sym}_trades.csv", mime="text/csv")
        else:
            st.write("No simulated trades executed for this ticker.")

        st.markdown("---")

# Place paper orders
if place_button:
    if not per_stock_data:
        st.warning("Run predictions first")
    else:
        st.warning("You are about to place MARKET orders on Alpaca PAPER account. Use Dry run first.")
        dry_run = st.checkbox("Dry run (do not submit)", value=True)
        confirm = st.checkbox("I confirm PAPER market orders", value=False)
        if confirm:
            placed = []
            for sym, info in per_stock_data.items():
                if "metadata" not in info: continue
                md = info["metadata"]
                if md["signal"] == "HOLD": continue
                side = "buy" if md["signal"] == "BUY" else "sell"
                qty = md.get("suggested_qty") or (int(fixed_qty) if qty_mode == "fixed_shares" else 1)
                if dry_run:
                    placed.append({"ticker": sym, "side": side, "qty": qty, "status": "dry_run", "suggested_entry_price": md.get("suggested_entry_price")})
                else:
                    try:
                        order = client.submit_order(symbol=sym, qty=qty, side=side, type="market", time_in_force="day")
                        placed.append({"ticker": sym, "side": side, "qty": qty, "alpaca_id": getattr(order,"id",None), "status": "submitted"})
                    except Exception as e:
                        placed.append({"ticker": sym, "error": str(e)})
            if placed:
                st.table(pd.DataFrame(placed))
            else:
                st.info("No orders created (no actionable signals).")

# Render help iframe overlay (height=400)
components.html(HELP_IFRAME_HTML, height=400, scrolling=False)

st.markdown("---")
st.markdown("- Estimated profit uses historical positive/negative future returns where available; otherwise conservative fallback buffers are used. Test suggestions on Alpaca PAPER first.")


