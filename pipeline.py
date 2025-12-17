#!/usr/bin/env python3
"""
start_pipeline.py - Pipeline Coordinator
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from threading import Thread, Event
from dotenv import load_dotenv

# Add to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.database import Database

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pipeline')


class PipelineRunner:
    """Coordinates your existing scripts in scripts/spot/"""
    
    def __init__(self):
        self.db = Database()
        self.shutdown_event = Event()
        self.realtime_process = None
        self.gap_filler_thread = None
        
        # Paths to your existing scripts
        self.backfill_script = PROJECT_ROOT / 'scripts' / 'spot' / 'backfill.py'
        self.fill_gaps_script = PROJECT_ROOT / 'scripts' / 'spot' / 'fill_gaps.py'
        self.realtime_script = PROJECT_ROOT / 'scripts' / 'spot' / 'realtime_collector.py'
        
        # Config
        self.pairs = os.getenv('PAIRS', 'BTCUSDT,ETHUSDT').split(',')
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        self.gap_check_hours = int(os.getenv('GAP_CHECK_HOURS', '1'))
    
    def start(self):
        """Start the complete pipeline"""
        print("\n" + "="*70)
        print("🚀 CRYPTO DATA PIPELINE")
        print("="*70)
        print(f"Pairs: {', '.join(self.pairs)}")
        print("="*70 + "\n")
        
        # Step 1: Test systems
        print("🔧 Testing connections...")
        if not self.db.test_connection():
            print("❌ Database connection failed!")
            return
        print("✅ Database connected\n")
        
        # Step 2: Check database status
        print("📊 Checking database status...")
        is_empty = self.db.is_database_empty()
        
        if is_empty:
            print("⚠️  Database is empty")
            print("\nOptions:")
            print("1. Run full backfill (download 8 years of data)")
            print("2. Skip and collect from now onwards")
            
            choice = input("\nSelect (1/2): ").strip()
            
            if choice == '1':
                print()
                self.run_backfill()
            else:
                print("✅ Skipping backfill\n")
        else:
            stats = self.db.get_statistics()
            print(f"✅ Database has {stats['total_rows']:,} rows")
            print(f"   Size: {stats['db_size']}\n")
        
        # Step 3: Automatically check and fill gaps
        print("🔍 Checking for recent gaps...")
        self.fill_initial_gaps()
        
        # Step 4: Start continuous collection
        print("\n" + "="*70)
        print("🚀 STARTING CONTINUOUS COLLECTION")
        print("="*70)
        print("This will:")
        print(f"  • Collect real-time data every minute")
        print(f"  • Check for gaps every {self.gap_check_hours} hour(s)")
        print(f"  • Auto-restart if any component fails")
        print("="*70)
        
        start = input("\nStart pipeline? (y/n): ").strip().lower()
        if start != 'y':
            print("Cancelled.")
            return
        
        print()
        
        # Start components
        self.start_realtime()
        self.start_gap_filler()
        
        print("\n✅ Pipeline running!")
        print("="*70)
        print("📊 Monitor logs: tail -f logs/pipeline.log")
        print("🛑 Stop: Press Ctrl+C")
        print("="*70 + "\n")
        
        # Keep alive and monitor
        try:
            while not self.shutdown_event.is_set():
                # Check if realtime died
                if self.realtime_process and self.realtime_process.poll() is not None:
                    logger.error("❌ Real-time collector died! Restarting...")
                    print("⚠️  Real-time collector restarted")
                    self.start_realtime()
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
        finally:
            self.shutdown()
    
    def run_backfill(self):
        """Run your existing backfill.py"""
        logger.info("Running historical backfill...")
        print("="*70)
        print("📥 HISTORICAL BACKFILL")
        print("="*70)
        print("This may take 1-4 hours depending on your connection.")
        print("You can monitor progress in logs/backfill.log")
        print("="*70 + "\n")
        
        try:
            # Run backfill script
            subprocess.run([sys.executable, str(self.backfill_script)])
            print("\n✅ Backfill completed\n")
            
        except KeyboardInterrupt:
            print("\n⚠️  Backfill interrupted\n")
        except Exception as e:
            logger.error(f"Backfill error: {e}")
            print(f"❌ Backfill error: {e}\n")
    
    def run_quick_gap_check(self):
        """Run quick gap check (last 50 gaps)"""
        logger.info("Running quick gap check...")
        print("Checking for gaps... (this may take a minute)")
        
        try:
            # Run fill_gaps with option 3 (first 50 gaps)
            result = subprocess.run(
                [sys.executable, str(self.fill_gaps_script)],
                input='3\n',  # Select option 3
                text=True,
                capture_output=True,
                timeout=300
            )
            
            if 'No gaps found' in result.stdout:
                print("✅ No gaps found\n")
            else:
                print("✅ Gap check completed (see logs/fill_gaps.log)\n")
                
        except subprocess.TimeoutExpired:
            print("⚠️  Gap check timed out\n")
        except Exception as e:
            logger.error(f"Gap check error: {e}")
            print(f"⚠️  Gap check error: {e}\n")
    
    def start_realtime(self):
        """Start your existing realtime_collector.py"""
        try:
            self.realtime_process = subprocess.Popen(
                [sys.executable, str(self.realtime_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            logger.info(f"Real-time collector started (PID: {self.realtime_process.pid})")
            print(f"✅ Real-time collector started (PID: {self.realtime_process.pid})")
            
        except Exception as e:
            logger.error(f"Failed to start real-time collector: {e}")
            print(f"❌ Failed to start real-time collector: {e}")
    
    def start_gap_filler(self):
        """Start periodic gap checking thread"""
        self.gap_filler_thread = Thread(
            target=self._gap_filler_loop,
            daemon=True,
            name="GapFiller"
        )
        self.gap_filler_thread.start()
        logger.info(f"Periodic gap checker started (every {self.gap_check_hours}h)")
        print(f"✅ Periodic gap checker started (every {self.gap_check_hours}h)")
    
    def fill_initial_gaps(self):
        """Fill gaps on startup (automatic, no prompts)"""
        from scripts.spot.binance_api import BinanceAPI
        
        logger.info("Running initial gap check...")
        print("Checking for gaps in last 24 hours...")
        
        api = BinanceAPI()
        
        for pair in self.pairs:
            try:
                # Check last 24 hours
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=1)
                
                gaps = self.db.get_data_gaps(
                    pair=pair,
                    timeframe=self.timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    min_gap_minutes=2
                )
                
                if not gaps:
                    print(f"  ✅ {pair}: No gaps found")
                    continue
                
                print(f"  ⚠️  {pair}: Found {len(gaps)} gaps - filling...")
                
                total_filled = 0
                for gap_start, gap_end in gaps[:20]:  # First 20 gaps
                    try:
                        candles = api.fetch_historical_range(
                            symbol=pair,
                            interval=self.timeframe,
                            start_date=gap_start,
                            end_date=gap_end
                        )
                        
                        if candles:
                            inserted = self.db.insert_candles(candles)
                            total_filled += inserted
                    except Exception as e:
                        logger.error(f"Error filling gap: {e}")
                
                if total_filled > 0:
                    print(f"  ✅ {pair}: Filled {total_filled} candles")
                else:
                    print(f"  ✅ {pair}: Gaps already filled")
                    
            except Exception as e:
                logger.error(f"Error checking gaps for {pair}: {e}")
                print(f"  ⚠️  {pair}: Error checking gaps")
        
        print()
    
    def _gap_filler_loop(self):
        """Background thread for periodic gap checking"""
        # Import API here (after path is set)
        from scripts.spot.binance_api import BinanceAPI
        api = BinanceAPI()
        
        # Wait before first check
        wait_minutes = 10
        logger.info(f"Gap checker: Waiting {wait_minutes} minutes before first check...")
        self.shutdown_event.wait(wait_minutes * 60)
        
        while not self.shutdown_event.is_set():
            try:
                logger.info("="*70)
                logger.info(f"PERIODIC GAP CHECK - {datetime.now(timezone.utc)}")
                logger.info("="*70)
                
                # Fill gaps directly (no subprocess)
                self.fill_gaps_internal(api)
                
                logger.info(f"Gap check complete. Next check in {self.gap_check_hours}h")
                logger.info("="*70)
                
                # Wait for next interval
                self.shutdown_event.wait(self.gap_check_hours * 3600)
                
            except Exception as e:
                logger.error(f"Gap filler loop error: {e}")
                self.shutdown_event.wait(300)  # Wait 5 min on error
    
    def fill_gaps_internal(self, api):
        """Fill gaps directly without calling external script"""
        for pair in self.pairs:
            try:
                logger.info(f"Checking gaps for {pair}...")
                
                # Check last 24 hours for recent gaps
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=1)
                
                gaps = self.db.get_data_gaps(
                    pair=pair,
                    timeframe=self.timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    min_gap_minutes=2
                )
                
                if not gaps:
                    logger.info(f"  ✅ {pair}: No gaps found")
                    continue
                
                logger.info(f"  ⚠️  {pair}: Found {len(gaps)} gaps")
                
                total_filled = 0
                for gap_start, gap_end in gaps[:50]:  # Limit to 50 gaps
                    gap_duration = gap_end - gap_start
                    logger.info(f"  Filling: {gap_start} → {gap_end} ({gap_duration})")
                    
                    try:
                        # Fetch missing candles
                        candles = api.fetch_historical_range(
                            symbol=pair,
                            interval=self.timeframe,
                            start_date=gap_start,
                            end_date=gap_end
                        )
                        
                        if candles:
                            inserted = self.db.insert_candles(candles)
                            total_filled += inserted
                            logger.info(f"  ✅ Filled {inserted} candles")
                        else:
                            logger.warning(f"  ⚠️  No data available for this gap")
                    
                    except Exception as e:
                        logger.error(f"  ❌ Failed to fill gap: {e}")
                
                logger.info(f"✅ {pair}: Total {total_filled} candles filled")
                
            except Exception as e:
                logger.error(f"❌ Error filling gaps for {pair}: {e}")
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down pipeline...")
        print("\n🛑 Shutting down...\n")
        
        self.shutdown_event.set()
        
        # Stop realtime collector
        if self.realtime_process:
            logger.info("Stopping real-time collector...")
            try:
                self.realtime_process.terminate()
                self.realtime_process.wait(timeout=10)
                print("✅ Real-time collector stopped")
            except:
                self.realtime_process.kill()
                print("⚠️  Real-time collector force killed")
        
        # Wait for gap filler
        if self.gap_filler_thread and self.gap_filler_thread.is_alive():
            self.gap_filler_thread.join(timeout=5)
        
        self.db.close_all_connections()
        
        logger.info("Pipeline stopped")
        print("👋 Pipeline stopped\n")


def main():
    """Entry point"""
    # Ensure directories exist
    Path('logs').mkdir(exist_ok=True)
    Path('data/realtime_cache').mkdir(parents=True, exist_ok=True)
    
    runner = PipelineRunner()
    runner.start()


if __name__ == "__main__":
    main()