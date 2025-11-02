# robot.py
import time, os, yfinance as yf, alpaca_trade_api as tradeapi

api = tradeapi.REST(
        os.environ['PKUKDHXAYB4RJBDFHNZSZBF6HO'],
        os.environ['FNdgbsu3SM9LNQcrUfcPhA6Y5j2M599dTEnf9pi9kaRh'],
        'https://paper-api.alpaca.markets')

while True:
    data = yf.download("AAPL", period="5d")
    avg5 = data['Close'][-5:].mean()
    yesterday = data['Close'][-1]
    side = "buy" if yesterday > avg5 else "sell"

    api.submit_order("AAPL", 1, side, 'market', 'day')
    print("Ordered", side)
    time.sleep(86400)   # wait 24 hours


from datetime import datetime
import requests

msg = f"{datetime.now():%d %b}  Reliance Bot: {side.upper()} 1 share"
requests.post(
    "https://api.callmebot.com/whatsapp.php",
    params={"phone": "+919010962805", "text": msg, "apikey": "demo"})