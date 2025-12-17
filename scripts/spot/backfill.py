"""
backfill.py - Historical Data Backfill Script
Downloads historical OHLCV data from Binance and stores in TimescaleDB
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.database import Database
from binance_api import BinanceAPI
from datetime import datetime, timedelta, timezone
import time
import logging
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv('config/.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backfill.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HistoricalBackfill:
    """Handles historical data backfill operations"""
    
    def __init__(self):
        """Initialize database and Binance API connections"""
        logger.info("Initializing Historical Backfill...")
        self.db = Database()
        self.binance = BinanceAPI()
        
        # Configuration
        self.pairs = os.getenv('PAIRS', 'BTCUSDT,ETHUSDT').split(',')
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        self.start_date_str = os.getenv('START_DATE', '2017-08-17')
        
        # Parse start date
        self.start_date = datetime.strptime(self.start_date_str, '%Y-%m-%d')
        self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        self.end_date = datetime.now(timezone.utc)
        
        logger.info(f"Pairs: {self.pairs}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
    
    def backfill_pair(self, pair):
        """
        Backfill historical data for a single trading pair
        
        Args:
            pair: Trading pair symbol (e.g., 'BTCUSDT')
        """
        logger.info(f"Starting backfill for {pair}")
        
        try:
            # Check if pair already has data
            last_candle_time = self.db.get_last_candle_time(pair, self.timeframe)
            
            if last_candle_time:
                logger.info(f"{pair} already has data up to {last_candle_time}")
                user_input = input(f"Continue from {last_candle_time}? (y/n): ")
                if user_input.lower() != 'y':
                    logger.info(f"Skipping {pair}")
                    return
                # Convert to UTC if not already
                if last_candle_time.tzinfo is None:
                    start_date = last_candle_time.replace(tzinfo=timezone.utc) + timedelta(minutes=1)
                else:
                    start_date = last_candle_time.astimezone(timezone.utc) + timedelta(minutes=1)
            else:
                start_date = self.start_date
            
            # Verify symbol exists on Binance
            if not self.binance.verify_symbol_exists(pair):
                logger.error(f"{pair} does not exist on Binance")
                return
            
            # # Fetch historical data
            # logger.info(f"Fetching {pair} data from {start_date} to {self.end_date}")
            # candles = self.binance.fetch_historical_range(
            #     symbol=pair,
            #     interval=self.timeframe,
            #     start_date=start_date,
            #     end_date=self.end_date
            # )
            
            # if not candles:
            #     logger.warning(f"No data fetched for {pair}")
            #     return
            # Fetch historical data
            logger.info(f"Fetching {pair} data from {start_date} to {self.end_date}")

            # Try fetching from start_date first
            candles = self.binance.fetch_historical_range(
                symbol=pair,
                interval=self.timeframe,
                start_date=start_date,
                end_date=self.end_date
            )

            # If no data, there might be a gap - try fetching latest available
            if not candles or len(candles) == 0:
                logger.warning(f"No data from {start_date}, checking for gaps...")
                
                # Fetch just the latest 2 candles to see where data resumes
                test_candles = self.binance.fetch_klines(pair, self.timeframe, limit=2)
                
                if test_candles:
                    first_available = test_candles[0][0]
                    # Ensure timezone-aware
                    if first_available.tzinfo is None:
                        first_available = first_available.replace(tzinfo=timezone.utc)
                    logger.info(f"Found data gap. Data resumes at {first_available}")
                    
                    # Fetch from where data actually exists
                    candles = self.binance.fetch_historical_range(
                        symbol=pair,
                        interval=self.timeframe,
                        start_date=first_available,
                        end_date=self.end_date
                    )
                
                if not candles:
                    logger.error(f"No data available for {pair} from {start_date} onwards")
                    return
            
            # Insert into database in batches
            batch_size = 5000
            total_inserted = 0
            
            for i in range(0, len(candles), batch_size):
                batch = candles[i:i + batch_size]
                inserted = self.db.insert_candles(batch)
                total_inserted += inserted
                
                logger.info(f"Progress: {i + len(batch)}/{len(candles)} candles processed")
                time.sleep(0.1)  # Small delay to not overwhelm database
            
            logger.info(f"✅ Backfill complete for {pair}: {total_inserted} new candles inserted")
            
        except Exception as e:
            logger.error(f"Failed to backfill {pair}: {e}")
            raise
    
    def backfill_all(self):
        """Backfill all configured trading pairs"""
        logger.info("="*60)
        logger.info("Starting Historical Data Backfill")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            for pair in self.pairs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing: {pair}")
                logger.info(f"{'='*60}")
                
                self.backfill_pair(pair)
                
                # Small delay between pairs
                time.sleep(1)
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ Backfill Complete!")
            logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
            logger.info(f"{'='*60}")
            
            # Show statistics
            self.show_statistics()
            
        except KeyboardInterrupt:
            logger.warning("\nBackfill interrupted by user")
            self.show_statistics()
        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            raise
        # finally:
        #     self.db.close_all_connections()
    
    def show_statistics(self):
        """Display database statistics after backfill"""
        logger.info("\n" + "="*60)
        logger.info("Database Statistics")
        logger.info("="*60)
        
        stats = self.db.get_statistics()
        if stats:
            logger.info(f"Total rows: {stats['total_rows']:,}")
            logger.info(f"Database size: {stats['db_size']}")
            
            logger.info("\nData by pair and timeframe:")
            for row in stats['stats']:
                pair, timeframe, count, first, last = row
                logger.info(f"  {pair} {timeframe}: {count:,} candles ({first} to {last})")
    
    def verify_data_integrity(self, pair):
        """
        Check for gaps in historical data
        
        Args:
            pair: Trading pair to verify
        """
        logger.info(f"Verifying data integrity for {pair}")
        
        gaps = self.db.get_data_gaps(
            pair=pair,
            timeframe=self.timeframe,
            start_time=self.start_date,
            end_time=self.end_date
        )
        
        if gaps:
            logger.warning(f"Found {len(gaps)} gaps in {pair} data:")
            for gap_start, gap_end in gaps:
                duration = gap_end - gap_start
                logger.warning(f"  Gap: {gap_start} to {gap_end} (duration: {duration})")
        else:
            logger.info(f"✅ No significant gaps found in {pair} data")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("Historical Data Backfill Tool")
    print("="*60)
    print("\nThis will download historical OHLCV data from Binance")
    print("and store it in your TimescaleDB database.")
    print("\nConfiguration:")
    print(f"  Pairs: {os.getenv('PAIRS', 'BTCUSDT,ETHUSDT')}")
    print(f"  Timeframe: {os.getenv('TIMEFRAME', '1m')}")
    print(f"  Start Date: {os.getenv('START_DATE', '2017-08-17')}")
    print(f"  End Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("\n" + "="*60)
    
    # Confirm before proceeding
    response = input("\nProceed with backfill? (y/n): ")
    if response.lower() != 'y':
        print("Backfill cancelled.")
        return
    
    # Run backfill
    backfill = HistoricalBackfill()
    backfill.backfill_all()
    
    # Optional: Verify data integrity
    verify = input("\nVerify data integrity? (y/n): ")
    if verify.lower() == 'y':
        for pair in backfill.pairs:
            backfill.verify_data_integrity(pair)
            
    # Close connections at the very end
    backfill.db.close_all_connections()


if __name__ == "__main__":
    main()