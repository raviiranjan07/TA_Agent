#!/usr/bin/env python3
"""
archive_download.py – Binance Historical Data Archive Downloader
Downloads *only* missing monthly ZIPs from data.binance.vision
and bulk-inserts them into PostgreSQL (ON CONFLICT DO NOTHING).

Features
--------
* True smart back-fill (starts after the last candle in DB)
* Local ZIP cache → no re-download
* Streaming CSV parsing (low memory)
* tqdm progress bar
* CLI: --symbol BTCUSDT,ETHUSDT --interval 1m
* Robust error handling
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path

import requests
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Adjust path so `database.py` in the same folder is importable
sys.path.append(str(Path(__file__).parent))
from scripts.database import Database  # type: ignore

# --------------------------------------------------------------------------- #
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/archive_download.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
CACHE_DIR = Path("data/archive_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
class BinanceArchiveDownloader:
    def __init__(self) -> None:
        self.db = Database()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "binance-archive-downloader/1.0"})

    # ------------------------------------------------------------------- #
    @staticmethod
    def _first_of_month(dt: datetime) -> datetime:
        """Return 00:00:00 UTC of the first day of the month."""
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # ------------------------------------------------------------------- #
    def _months_range(self, start: datetime, end: datetime) -> list[tuple[int, int]]:
        """Yield (year, month) tuples from start (inclusive) to end (inclusive)."""
        cur = self._first_of_month(start)
        months: list[tuple[int, int]] = []
        while cur <= end:
            months.append((cur.year, cur.month))
            # next month
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return months

    # ------------------------------------------------------------------- #
    def _download_zip(
        self, symbol: str, interval: str, year: int, month: int
    ) -> Path | None:
        """Download (if needed) and return cached ZIP path."""
        filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
        cache_path = CACHE_DIR / filename

        if cache_path.exists():
            log.debug("Cache hit: %s", cache_path.name)
            return cache_path

        log.info("Downloading %s …", filename)
        try:
            r = self.session.get(url, timeout=60)
            if r.status_code == 404:
                log.warning("Not found on server: %s", filename)
                return None
            r.raise_for_status()
            cache_path.write_bytes(r.content)
            return cache_path
        except requests.RequestException as e:
            log.error("Download failed %s: %s", filename, e)
            return None

    # ------------------------------------------------------------------- #
    def _parse_zip(self, zip_path: Path) -> list[tuple]:
        """
        Return list of candle tuples:
        (datetime_UTC, symbol, interval, open, high, low, close,
         volume, quote_volume, num_trades, taker_buy_base, taker_buy_quote)
        """
        try:
            with zipfile.ZipFile(zip_path) as z:
                csv_name = zip_path.name.replace(".zip", ".csv")
                with z.open(csv_name) as f:
                    # Stream line-by-line → low memory
                    reader = csv.reader(TextIOWrapper(f, encoding="utf-8"))
                    candles = []
                    for row in reader:
                        if len(row) < 11:
                            continue  # malformed
                        ts = int(row[0]) // 1000
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        candle = (
                            dt,
                            zip_path.stem.split("-")[0],  # symbol
                            zip_path.stem.split("-")[1],  # interval
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(row[5]),
                            float(row[7]),
                            int(row[8]),
                            float(row[9]),
                            float(row[10]),
                        )
                        candles.append(candle)
                    log.debug("Parsed %d candles from %s", len(candles), zip_path.name)
                    return candles
        except Exception as e:
            log.error("Failed to parse %s: %s", zip_path.name, e)
            return []

    # ------------------------------------------------------------------- #
    def _import_month(self, candles: list[tuple]) -> int:
        """Insert one month in 10 k-row batches, return total inserted."""
        batch_size = 10_000
        inserted_total = 0
        for i in range(0, len(candles), batch_size):
            batch = candles[i : i + batch_size]
            inserted = self.db.insert_candles(batch)
            inserted_total += inserted
        return inserted_total

    # ------------------------------------------------------------------- #
    def download_and_import_range(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        months = self._months_range(start_dt, end_dt)
        log.info(
            "Will process %d month(s) for %s %s (%s → %s)",
            len(months),
            symbol,
            interval,
            start_dt.strftime("%Y-%m"),
            end_dt.strftime("%Y-%m"),
        )

        total_inserted = 0
        for year, month in tqdm(months, desc="Months", unit="month"):
            zip_path = self._download_zip(symbol, interval, year, month)
            if not zip_path:
                continue

            candles = self._parse_zip(zip_path)
            if not candles:
                continue

            inserted = self._import_month(candles)
            total_inserted += inserted
            log.info(
                "Month %04d-%02d → %6d candles inserted (%d total)",
                year,
                month,
                inserted,
                total_inserted,
            )

        log.info("Finished %s %s → %d new candles", symbol, interval, total_inserted)

    # ------------------------------------------------------------------- #
    def smart_backfill(self, symbol: str, interval: str = "1m") -> None:
        """Download only months that are missing in the DB."""
        log.info("\n=== Smart back-fill %s %s ===", symbol, interval)

        last_ts = self.db.get_last_candle_time(symbol, interval)
        if last_ts:
            # last candle is *inclusive* → start with the *next* month
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            else:
                last_ts = last_ts.astimezone(timezone.utc)

            start_dt = (last_ts + timedelta(days=32)).replace(day=1)
            log.info("DB already has data up to %s → resume from %s", last_ts, start_dt)
        else:
            start_dt = datetime(2017, 8, 1, tzinfo=timezone.utc)
            log.info("No data for %s → start from Binance launch %s", symbol, start_dt)

        # End = last complete month (yesterday)
        today = datetime.now(timezone.utc)
        end_dt = (today.replace(day=1) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if start_dt > end_dt:
            log.info("Nothing to download – DB is already up-to-date.")
            return

        self.download_and_import_range(symbol, interval, start_dt, end_dt)

    # ------------------------------------------------------------------- #
    def full_backfill(self, symbols: list[str], interval: str = "1m") -> None:
        """Run smart_backfill for every symbol."""
        log.info("\n=== FULL BACKFILL (%d symbols) ===", len(symbols))
        for sym in symbols:
            try:
                self.smart_backfill(sym, interval)
            except Exception as exc:
                log.exception("Back-fill failed for %s: %s", sym, exc)
        self.db.close_all_connections()


# --------------------------------------------------------------------------- #
def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance monthly archive → PostgreSQL (smart back-fill)"
    )
    parser.add_argument(
        "-s",
        "--symbol",
        action="append",
        required=True,
        help="Symbol(s) e.g. BTCUSDT (repeatable)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        default="1m",
        help="Kline interval (default: 1m)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore DB and download *everything* from 2017-08",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_cli()

    print("\n" + "=" * 60)
    print("Binance Historical Data Archive Downloader")
    print("=" * 60)

    if not args.full:
        confirm = input("\nRun **smart** back-fill (only missing months)? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return
    else:
        confirm = input("\nRun **full** back-fill (ignore DB, redownload all)? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    downloader = BinanceArchiveDownloader()

    if args.full:
        # Force start from Binance launch
        start = datetime(2017, 8, 1, tzinfo=timezone.utc)
        end = (datetime.now(timezone.utc).replace(day=1) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for sym in args.symbol:
            downloader.download_and_import_range(sym, args.interval, start, end)
    else:
        downloader.full_backfill(args.symbol, args.interval)

    print("\nAll done! Check logs/archive_download.log for details.\n")


if __name__ == "__main__":
    main()