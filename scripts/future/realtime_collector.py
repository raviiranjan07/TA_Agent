#!/usr/bin/env python3
"""
future/realtime_collector.py
- Listens to Binance Futures websocket kline_1m streams for configured pairs.
- On startup: flush local cache -> backfill missing candles -> start WS listeners.
- On DB failure: caches candles to disk and retries periodically.
"""

import asyncio
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import requests
import websockets
from dotenv import load_dotenv

from database import Database  # ensure Database class handles crypto_future_data table

# ---- Config ----
load_dotenv()
FUTURE_PAIRS = os.getenv("FUTURE_PAIRS", "BTCUSDT,ETHUSDT").split(",")
TIMEFRAME = os.getenv("FUTURE_TIMEFRAME", "1m")
WS_BASE = "wss://fstream.binance.com/ws"
REST_KLINES = "https://fapi.binance.com/fapi/v1/klines"

CACHE_DIR = Path("data/future_realtime_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FLUSH_INTERVAL = int(os.getenv("FUTURE_CACHE_FLUSH_INTERVAL", "30"))

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/future_realtime_collector.log"), logging.StreamHandler()]
)
logger = logging.getLogger("future_realtime_collector")

executor = ThreadPoolExecutor(max_workers=8)

# ---- Helpers ----
def utcnow():
    return datetime.now(timezone.utc)

def pair_socket(pair):
    return f"{pair.lower()}@kline_{TIMEFRAME}"

def cache_path_for(pair):
    return CACHE_DIR / f"{pair.lower()}_cache.json"

def read_cache(pair):
    if not cache_path_for(pair).exists():
        return []
    try:
        with cache_path_for(pair).open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read cache for {pair}: {e}")
        return []

def append_to_cache(pair, candle):
    p = cache_path_for(pair)
    data = read_cache(pair)
    data.append(candle)
    try:
        with p.open("w") as f:
            json.dump(data, f, default=str)
    except Exception as e:
        logger.error(f"Failed to write cache for {pair}: {e}")

def remove_cache(pair):
    p = cache_path_for(pair)
    if p.exists():
        try:
            p.unlink()
        except Exception as e:
            logger.error(f"Failed to delete cache {p}: {e}")

def kline_to_db_tuple(kline, pair):
    ts = datetime.fromtimestamp(int(kline[0])/1000, tz=timezone.utc).replace(tzinfo=None)
    return (
        ts, pair, TIMEFRAME,
        float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]),
        float(kline[5]), float(kline[7]), int(kline[8]),
        float(kline[9]), float(kline[10])
    )

def fetch_klines_rest(pair, start_dt, end_dt=None, limit=1000):
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
        resp = requests.get(REST_KLINES, params=params, timeout=30)
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        out.extend([kline_to_db_tuple(k, pair) for k in klines])
        last_ts = int(klines[-1][0])
        if end_dt and last_ts >= int(end_dt.timestamp() * 1000):
            break
        params["startTime"] = last_ts + 1
        if len(klines) < limit:
            break
        time.sleep(0.1)
    return out

def db_insert_candles_sync(db_obj, candles):
    return db_obj.insert_candles(candles, table="crypto_future_data")

def db_get_last_sync(db_obj, pair):
    return db_obj.get_last_candle_time(pair, TIMEFRAME, table="crypto_future_data")

# ---- Main Realtime Worker ----
class RealtimeCollector:
    def __init__(self, pairs):
        self.pairs = [p.upper() for p in pairs]
        self.db = Database()
        self.loop = asyncio.get_running_loop()
        self.ws_tasks = []
        self.flush_task = None
        self.running = True

    async def start(self):
        await self.flush_all_caches()
        await self.backfill_missing()
        self.flush_task = asyncio.create_task(self.periodic_flush_cache())
        for p in self.pairs:
            self.ws_tasks.append(asyncio.create_task(self.run_pair_ws(p)))
        await asyncio.gather(*self.ws_tasks)

    async def backfill_missing(self):
        logger.info("Starting backfill check for future pairs...")
        for pair in self.pairs:
            try:
                last = await self.loop.run_in_executor(executor, db_get_last_sync, self.db, pair)
                start_dt = utcnow() - timedelta(days=30) if last is None else last.replace(tzinfo=timezone.utc) + timedelta(minutes=1)
                end_dt = utcnow()
                if start_dt >= end_dt:
                    logger.info(f"No backfill required for {pair}")
                    continue
                candles = fetch_klines_rest(pair, start_dt, end_dt)
                if candles:
                    inserted = await self.loop.run_in_executor(executor, db_insert_candles_sync, self.db, candles)
                    logger.info(f"Backfill: inserted {inserted} candles for {pair}")
            except Exception as e:
                logger.error(f"Error during backfill for {pair}: {e}")

    async def periodic_flush_cache(self):
        while self.running:
            try:
                await self.flush_all_caches()
            except Exception as e:
                logger.error(f"Cache flush error: {e}")
            await asyncio.sleep(FLUSH_INTERVAL)

    async def flush_all_caches(self):
        for pair in self.pairs:
            cached = read_cache(pair)
            if not cached:
                continue
            to_insert = []
            for item in cached:
                if isinstance(item, dict):
                    ts = datetime.fromisoformat(item["time"])
                    dbtuple = (
                        ts.replace(tzinfo=None),
                        pair, TIMEFRAME,
                        float(item["open"]), float(item["high"]), float(item["low"]), float(item["close"]),
                        float(item.get("volume", 0)), float(item.get("quote_volume", 0)), int(item.get("num_trades", 0)),
                        float(item.get("taker_buy_base", 0)), float(item.get("taker_buy_quote", 0))
                    )
                    to_insert.append(dbtuple)
                elif isinstance(item, (list, tuple)):
                    to_insert.append(tuple(item))
            if to_insert:
                try:
                    inserted = await self.loop.run_in_executor(executor, db_insert_candles_sync, self.db, to_insert)
                    logger.info(f"Flushed cache for {pair}: inserted {inserted} candles. Deleting cache file.")
                    remove_cache(pair)
                except Exception as e:
                    logger.warning(f"Failed to flush cache for {pair}: {e}")

    async def run_pair_ws(self, pair):
        url = f"{WS_BASE}/{pair_socket(pair)}"
        backoff = 1
        max_backoff = 120
        while True:
            try:
                logger.info(f"Connecting WS for {pair} -> {url}")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info(f"Connected WS for {pair}")
                    backoff = 1
                    async for raw in ws:
                        msg = json.loads(raw)
                        k = msg.get("k")
                        if not k or not k.get("x"):
                            continue
                        ts = datetime.fromtimestamp(int(k["t"])/1000, tz=timezone.utc).replace(tzinfo=None)
                        db_tuple = (
                            ts, pair, TIMEFRAME,
                            float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]),
                            float(k["v"]), float(k.get("q", 0)), int(k.get("n", 0)),
                            float(k.get("V", 0)), float(k.get("Q", 0))
                        )
                        try:
                            await self.loop.run_in_executor(executor, db_insert_candles_sync, self.db, [db_tuple])
                        except Exception:
                            cache_item = {
                                "time": ts.isoformat(),
                                "open": k["o"], "high": k["h"], "low": k["l"], "close": k["c"],
                                "volume": k["v"], "quote_volume": k.get("q", 0), "num_trades": k.get("n", 0),
                                "taker_buy_base": k.get("V", 0), "taker_buy_quote": k.get("Q", 0)
                            }
                            append_to_cache(pair, cache_item)
            except Exception as e:
                logger.warning(f"WS error for {pair}: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff*2, max_backoff)

# ---- Entrypoint ----
def main():
    logger.info("Starting Futures Realtime Collector")
    rc = RealtimeCollector(FUTURE_PAIRS)
    try:
        asyncio.run(rc.start())
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
    except Exception as e:
        logger.error(f"Collector crashed: {e}")

if __name__ == "__main__":
    main()
