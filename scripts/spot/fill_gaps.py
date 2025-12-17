"""
============================================================
SMART GAP FILLER - FINAL VERSION (BULLETPROOF)
Detects gaps → Fills via REST API or ZIP Archive
============================================================
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from binance.client import Client
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.database import Database
from dotenv import load_dotenv
import requests
import zipfile
import csv
from pathlib import Path

load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise Exception("Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")

# CONFIG
PAIR = "BTCUSDT"
TIMEFRAME = "1m"
MAX_CANDLES_PER_REQUEST = 1000
ARCHIVE_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
CACHE_DIR = Path("data/archive_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
}
BINANCE_INTERVAL = TIMEFRAME_MAP[TIMEFRAME]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/smart_fill_gaps.log"), logging.StreamHandler()]
)
logger = logging.getLogger("__main__")

db = Database()
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def transform_candle(candle):
    """Convert Binance kline → DB tuple (NAIVE UTC)"""
    dt = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
    return (
        dt.replace(tzinfo=None),  # ← NAIVE UTC
        PAIR, TIMEFRAME,
        float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]),
        float(candle[5]), float(candle[7]), int(candle[8]),
        float(candle[9]), float(candle[10])
    )


def fetch_via_rest(start_time, end_time):
    """Fetch via Binance REST API → NAIVE UTC"""
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    all_klines = []
    while start_ts < end_ts:
        klines = client.get_klines(
            symbol=PAIR, interval=BINANCE_INTERVAL,
            startTime=start_ts, endTime=end_ts,
            limit=MAX_CANDLES_PER_REQUEST
        )
        if not klines:
            break
        all_klines.extend([transform_candle(k) for k in klines])
        start_ts = klines[-1][0] + 1
    logger.info(f"REST API: Fetched {len(all_klines)} candles")
    return all_klines


def download_zip_archive(year, month):
    """Download and parse ZIP → return NAIVE UTC candles"""
    filename = f"{PAIR}-{TIMEFRAME}-{year}-{month:02d}.zip"
    url = f"{ARCHIVE_BASE_URL}/{PAIR}/{TIMEFRAME}/{filename}"
    cache_path = CACHE_DIR / filename

    if cache_path.exists():
        logger.info(f"Using cached: {filename}")
    else:
        logger.info(f"Downloading archive: {filename}")
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 404:
                logger.warning(f"Archive not found: {filename}")
                return None
            response.raise_for_status()
            cache_path.write_bytes(response.content)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    try:
        with zipfile.ZipFile(cache_path) as z:
            csv_name = filename.replace('.zip', '.csv')
            with z.open(csv_name) as f:
                reader = csv.reader(f.read().decode('utf-8').splitlines())
                candles = []
                for row in reader:
                    dt = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
                    candle = (
                        dt.replace(tzinfo=None),  # ← NAIVE UTC
                        PAIR, TIMEFRAME,
                        float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                        float(row[5]), float(row[7]), int(row[8]),
                        float(row[9]), float(row[10])
                    )
                    candles.append(candle)
                logger.info(f"Archive: Parsed {len(candles)} candles")
                return candles
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        return None


def fetch_via_archive(gap_start, gap_end):
    candles = []
    gap_start_utc = gap_start.astimezone(timezone.utc).replace(tzinfo=None)
    gap_end_utc = gap_end.astimezone(timezone.utc).replace(tzinfo=None)

    current = gap_start.replace(day=1)
    while True:
        if current > gap_end:
            break
        year, month = current.year, current.month
        month_candles = download_zip_archive(year, month)
        if month_candles:
            filtered = [
                c for c in month_candles
                if gap_start_utc <= c[0] <= gap_end_utc
            ]
            candles.extend(filtered)
            logger.info(f"Collected {len(filtered)} from {year}-{month:02d}")
        else:
            logger.warning(f"Missing archive: {year}-{month:02d}")

        # Next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    logger.info(f"Archive: Total {len(candles)} candles for gap {gap_start} → {gap_end}")
    return candles


def is_recent_gap(gap_start):
    """Gap is recent if start >= 2022"""
    return gap_start.year >= 2022


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("SMART GAP FILLER - AUTO ROUTING")
    logger.info("="*60)

    print("1. Fill ALL gaps (smart routing)\n2. Fill 2025 gaps\n3. First 50 gaps")
    option = input("Select (1/2/3): ").strip()

    # === DETECT ALL GAPS ===
    full_start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    full_end = datetime.now(timezone.utc) + timedelta(days=1)
    gaps = db.get_data_gaps(PAIR, TIMEFRAME, full_start, full_end)
    logger.info(f"Found {len(gaps)} gaps")

    if option == "2":
        gaps = [g for g in gaps if g[0].year >= 2025]
    elif option == "3":
        gaps = gaps[:50]

    logger.info(f"Processing {len(gaps)} gaps")

    for idx, (gap_start, gap_end) in enumerate(gaps, 1):
        logger.info(f"Gap {idx}/{len(gaps)}: {gap_start} → {gap_end}")

        if is_recent_gap(gap_start):
            logger.info("Using REST API (recent gap)")
            candles = fetch_via_rest(gap_start, gap_end)
        else:
            logger.info("Using ZIP Archive (historical gap)")
            candles = fetch_via_archive(gap_start, gap_end)

        if candles:
            inserted = db.insert_candles(candles)
            logger.info(f"Inserted {inserted} new candles")
        else:
            logger.warning("No data fetched")

    logger.info("SMART GAP FILLING COMPLETE!")
    db.close_all_connections()