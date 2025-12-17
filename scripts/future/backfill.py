import asyncio
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from future.binance_api import fetch_futures_klines
from scripts.database import Database

executor = ThreadPoolExecutor(max_workers=8)
TIMEFRAME = "1m"
BACKFILL_LIMIT = 1000
TABLE_NAME = "crypto_future_data"

db = Database()

async def backfill_pair(pair):
    last_candle = db.get_last_candle_time(pair, TIMEFRAME, table=TABLE_NAME)
    if last_candle is None:
        start_dt = datetime.now(timezone.utc) - timedelta(days=7)
    else:
        start_dt = last_candle + timedelta(minutes=1)
    
    end_dt = datetime.now(timezone.utc)
    if start_dt >= end_dt:
        print(f"No backfill needed for {pair}")
        return

    klines = await asyncio.get_running_loop().run_in_executor(
        executor, fetch_futures_klines, pair, start_dt, end_dt, BACKFILL_LIMIT
    )

    db_tuples = []
    for k in klines:
        ts = datetime.fromtimestamp(k[0]/1000, tz=timezone.utc).replace(tzinfo=None)
        db_tuples.append((
            ts, pair, TIMEFRAME,
            float(k[1]), float(k[2]), float(k[3]), float(k[4]),
            float(k[5]), float(k[7]), int(k[8]),
            float(k[9]), float(k[10])
        ))
    
    if db_tuples:
        inserted = db.insert_candles(db_tuples, table=TABLE_NAME)
        print(f"Inserted {inserted} candles for {pair}")
