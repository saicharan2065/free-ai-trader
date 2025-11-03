import os, time, datetime as dt
import yfinance as yf, alpaca_trade_api as tradeapi

api = tradeapi.REST(os.environ['PKUKDHXAYB4RJBDFHNZSZBF6HO'],
                    os.environ['FNdgbsu3SM9LNQcrUfcPhA6Y5j2M599dTEnf9pi9kaRh'],
                    base_url='https://paper-api.alpaca.markets')

def get_signal():
    data = yf.download("AAPL", period="10d", progress=False)
    wd = data[data.index.weekday < 5]          # drop weekends
    if len(wd) < 2:
        return "HOLD"
    yesterday = wd['Close'].iloc[-1]
    avg5      = wd['Close'].iloc[-5:].mean()
    return "BUY" if yesterday > avg5 else "SELL"

if __name__ == "__main__":
    while True:
        side = get_signal()
        if side != "HOLD":
            api.submit_order(symbol="AAPL", qty=1, side=side,
                             type="market", time_in_force="day")
        print(dt.datetime.now(), "Signal:", side)
        time.sleep(86400)          # 1 day
