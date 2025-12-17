import os
import requests
from dotenv import load_dotenv

load_dotenv()

FUTURES_REST = "https://fapi.binance.com/fapi/v1/klines"  # USDT-M futures endpoint

TIMEFRAME = os.getenv("TIMEFRAME", "1m")

def fetch_futures_klines(pair, start_dt, end_dt=None, limit=1000):
    """
    Fetch USDT-margined perpetual futures klines from Binance.
    Returns list of klines (list format, same as spot).
    """
    params = {
        "symbol": pair,
        "interval": TIMEFRAME,
        "limit": limit,
        "startTime": int(start_dt.timestamp() * 1000)
    }
    if end_dt:
        params["endTime"] = int(end_dt.timestamp() * 1000)
    
    out = []
    while True:
        resp = requests.get(FUTURES_REST, params=params, timeout=30)
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        out.extend(klines)
        last_ts = int(klines[-1][0])
        if end_dt and last_ts >= int(end_dt.timestamp() * 1000):
            break
        params["startTime"] = last_ts + 1
        if len(klines) < limit:
            break
    return out
