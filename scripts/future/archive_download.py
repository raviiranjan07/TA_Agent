import os
from pathlib import Path
from datetime import datetime, timezone
import json
from future.binance_api import fetch_futures_klines

ARCHIVE_DIR = Path("data/future_archive")
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

def save_future_klines(pair, start_dt, end_dt=None):
    klines = fetch_futures_klines(pair, start_dt, end_dt)
    filename = ARCHIVE_DIR / f"{pair}_{start_dt.date()}_{end_dt.date() if end_dt else 'now'}.json"
    with filename.open("w") as f:
        json.dump(klines, f, default=str)
    print(f"Saved {len(klines)} candles to {filename}")
