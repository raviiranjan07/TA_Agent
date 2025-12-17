"""
binance_api.py - Binance API Wrapper
Handles all Binance API interactions for fetching historical and real-time data
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BinanceAPI:
    """Wrapper for Binance API operations"""
    
    # Binance timeframe mapping
    INTERVAL_MAP = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH,
    }
    
    def __init__(self):
        """Initialize Binance client"""
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        
        # API keys are optional for public endpoints
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
            logger.info("Binance client initialized with API keys")
        else:
            self.client = Client()
            logger.info("Binance client initialized without API keys (public endpoints only)")
        
        self.max_candles_per_request = 1000
        self.rate_limit_delay = 0.1  # seconds between requests
    
    def verify_symbol_exists(self, symbol: str) -> bool:
        """
        Check if a trading pair exists on Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            bool: True if symbol exists
        """
        try:
            info = self.client.get_symbol_info(symbol)
            return info is not None
        except BinanceAPIException as e:
            logger.error(f"Symbol {symbol} not found: {e}")
            return False
    
    def fetch_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Tuple]:
        """
        Fetch klines (candlestick data) from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '1h', '1d')
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC)
            limit: Number of candles to fetch (max 1000)
            
        Returns:
            List of tuples ready for database insertion
        """
        try:
            # Get Binance interval
            binance_interval = self.INTERVAL_MAP.get(interval)
            if not binance_interval:
                raise ValueError(f"Invalid interval: {interval}")
            
            # Prepare parameters
            params = {
                'symbol': symbol,
                'interval': binance_interval,
                'limit': min(limit, self.max_candles_per_request)
            }
            
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            # Fetch klines
            klines = self.client.get_klines(**params)
            
            # Transform to database format
            candles = []
            for kline in klines:
                candle = self._transform_kline(kline, symbol, interval)
                candles.append(candle)
            
            return candles
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching {symbol} {interval}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return []
    
    def fetch_historical_range(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple]:
        """
        Fetch historical data for a date range (handles pagination)
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            start_date: Start datetime (UTC)
            end_date: End datetime (UTC)
            
        Returns:
            List of all candles in the range
        """
        logger.info(f"Fetching {symbol} {interval} from {start_date} to {end_date}")
        
        all_candles = []
        current_start = start_date
        
        while current_start < end_date:
            # Calculate end time for this batch
            batch_end = min(current_start + timedelta(hours=16), end_date)
            
            # Fetch batch
            candles = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=batch_end,
                limit=self.max_candles_per_request
            )
            
            if not candles:
                logger.warning(f"No data returned for {symbol} at {current_start}")
                # Try to skip ahead
                current_start = batch_end
                continue
            
            all_candles.extend(candles)
            
            # Update start time to last candle + 1 interval
            last_candle_time = candles[-1][0]
            current_start = last_candle_time + self._get_interval_delta(interval)
            
            # Log progress
            logger.info(f"Fetched {len(candles)} candles. Total: {len(all_candles)}")
            
            # Respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Safety check: if we got fewer candles than expected, we've reached the end
            if len(candles) < self.max_candles_per_request:
                break
        
        logger.info(f"Completed: {len(all_candles)} total candles fetched")
        return all_candles
    
    def fetch_latest_candle(self, symbol: str, interval: str) -> Optional[Tuple]:
        """
        Fetch the most recent completed candle
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            
        Returns:
            Single candle tuple or None
        """
        candles = self.fetch_klines(symbol, interval, limit=1)
        return candles[0] if candles else None
    
    def _transform_kline(self, kline: list, symbol: str, interval: str) -> Tuple:
        """
        Transform Binance kline to database tuple format
        
        Binance kline format:
        [
            0: Open time (ms)
            1: Open
            2: High
            3: Low
            4: Close
            5: Volume
            6: Close time (ms)
            7: Quote asset volume
            8: Number of trades
            9: Taker buy base asset volume
            10: Taker buy quote asset volume
            11: Ignore
        ]
        
        Database format:
        (time, pair, timeframe, open, high, low, close, volume,
         quote_volume, num_trades, taker_buy_base, taker_buy_quote)
        """
        # Convert timestamp to datetime (UTC, timezone-naive for PostgreSQL)
        timestamp = datetime.fromtimestamp(
            int(kline[0]) / 1000, 
            tz=timezone.utc
        ).replace(tzinfo=None)
        
        return (
            timestamp,                  # time
            symbol,                     # pair
            interval,                   # timeframe
            float(kline[1]),           # open
            float(kline[2]),           # high
            float(kline[3]),           # low
            float(kline[4]),           # close
            float(kline[5]),           # volume
            float(kline[7]),           # quote_volume
            int(kline[8]),             # num_trades
            float(kline[9]),           # taker_buy_base
            float(kline[10])           # taker_buy_quote
        )
    
    def _get_interval_delta(self, interval: str) -> timedelta:
        """Convert interval string to timedelta"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        elif unit == 'M':
            return timedelta(days=value * 30)  # Approximate
        else:
            return timedelta(minutes=1)  # Default
    
    def get_server_time(self) -> datetime:
        """Get Binance server time"""
        server_time = self.client.get_server_time()
        return datetime.fromtimestamp(
            server_time['serverTime'] / 1000,
            tz=timezone.utc
        )
    
    def test_connectivity(self) -> bool:
        """Test connection to Binance API"""
        try:
            self.client.ping()
            logger.info("Binance API connectivity: OK")
            return True
        except Exception as e:
            logger.error(f"Binance API connectivity failed: {e}")
            return False


# Test the API
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    api = BinanceAPI()
    
    # Test connectivity
    if api.test_connectivity():
        print("✅ Connected to Binance")
        
        # Test fetching latest candle
        candle = api.fetch_latest_candle('BTCUSDT', '1m')
        if candle:
            print(f"✅ Latest BTC 1m candle: {candle[0]} - Close: ${candle[7]}")
        
        # Test symbol verification
        if api.verify_symbol_exists('BTCUSDT'):
            print("✅ BTCUSDT exists on Binance")
    else:
        print("❌ Failed to connect to Binance")