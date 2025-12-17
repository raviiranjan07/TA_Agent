#!/usr/bin/env python3
"""
run.py - Master Orchestrator for Crypto Data Pipeline
Handles: Test → Backfill → Gap Fill → Real-time Collection → Feature Engineering
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from scripts.database import Database
from dotenv import load_dotenv
from scripts.spot.binance_api import BinanceAPI

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MasterOrchestrator:
    """Master control for the entire pipeline"""
    
    def __init__(self):
        """Initialize orchestrator"""
        self.db = Database()
        self.pairs = os.getenv('PAIRS', 'BTCUSDT,ETHUSDT').split(',')
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        
        # Ensure log directory exists
        Path('logs').mkdir(exist_ok=True)
    
    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "="*70)
        print(" 🚀 CRYPTO DATA PIPELINE - MASTER CONTROL")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   Pairs: {', '.join(self.pairs)}")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Database: {os.getenv('DB_NAME', 'crypto_data')}")
        print("\n" + "="*70 + "\n")
    
    def test_systems(self):
        """Test all system components"""
        logger.info("Testing system components...")
        print("\n🔧 Testing System Components...")
        
        # Test database
        print("\n1. Database Connection...")
        if not self.db.test_connection():
            print("   ❌ Database connection failed!")
            return False
        print("   ✅ Database connected")
        
        # Test Binance API
        print("\n2. Binance API...")
        try:
            api = BinanceAPI()
            if api.test_connectivity():
                print("   ✅ Binance API connected")
            else:
                print("   ❌ Binance API failed!")
                return False
        except Exception as e:
            print(f"   ❌ Binance API error: {e}")
            return False
        
        # Check table structure
        print("\n3. Database Schema...")
        try:
            stats = self.db.get_statistics()
            if stats:
                print(f"   ✅ Schema valid")
                print(f"   📊 Current data: {stats['total_rows']:,} rows")
                print(f"   💾 Size: {stats['db_size']}")
            else:
                print("   ⚠️  No statistics available")
        except Exception as e:
            print(f"   ❌ Schema check failed: {e}")
            return False
        
        # Check timeframe tables
        print("\n4. Timeframe Tables...")
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE 'btc_%'
                ORDER BY tablename
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['btc_5min', 'btc_15min', 'btc_1hr', 'btc_4hr', 'btc_1day']
            missing_tables = [t for t in expected_tables if t not in tables]
            
            if missing_tables:
                print(f"   ⚠️  Missing tables: {', '.join(missing_tables)}")
                print("   💡 Run Task 1.1 to create timeframe tables")
            else:
                print(f"   ✅ All timeframe tables exist")
            
            cursor.close()
        finally:
            self.db.return_connection(conn)
        
        print("\n✅ All systems operational!\n")
        return True
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*70)
        print(" 📋 MAIN MENU")
        print("="*70)
        print("\n1. 🔄 Historical Backfill (Download 8 years of data)")
        print("2. 🔍 Fill Data Gaps (Fix missing candles)")
        print("3. 📡 Start Real-time Collector (Live updates)")
        print("4. 📊 View Statistics")
        print("5. 🧪 Test Systems")
        print("6. 🚀 Full Auto-Run (Backfill → Gap Fill → Real-time)")
        print("7. 🧮 Feature Engineering (Calculate indicators)")
        print("8. 🛑 Exit")
        print("\n" + "="*70)
    
    def run_backfill(self):
        """Run historical backfill"""
        print("\n" + "="*70)
        print(" 🔄 HISTORICAL BACKFILL")
        print("="*70)
        
        confirm = input("\n⚠️  This will download ~8 years of data. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        
        logger.info("Starting historical backfill...")
        print("\n🚀 Starting backfill...\n")
        
        try:
            subprocess.run([sys.executable, 'scripts/spot/backfill.py'], check=True)
            print("\n✅ Backfill completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Backfill failed: {e}")
        except KeyboardInterrupt:
            print("\n⚠️  Backfill interrupted by user")
    
    def run_gap_filler(self):
        """Run gap filler"""
        print("\n" + "="*70)
        print(" 🔍 GAP FILLER")
        print("="*70)
        
        logger.info("Starting gap filler...")
        print("\n🔍 Checking for data gaps...\n")
        
        try:
            subprocess.run([sys.executable, 'scripts/spot/fill_gaps.py'], check=True)
            print("\n✅ Gap filling completed!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Gap filling failed: {e}")
        except KeyboardInterrupt:
            print("\n⚠️  Gap filling interrupted by user")
    
    def run_realtime(self):
        """Run real-time collector"""
        print("\n" + "="*70)
        print(" 📡 REAL-TIME COLLECTOR")
        print("="*70)
        
        logger.info("Starting real-time collector...")
        print("\n📡 Starting real-time data collection...")
        print("   (Press Ctrl+C to stop)\n")
        
        try:
            subprocess.run([sys.executable, 'scripts/spot/realtime_collector.py'], check=True)
        except KeyboardInterrupt:
            print("\n⚠️  Real-time collector stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Real-time collector failed: {e}")
    
    def show_statistics(self):
        """Display database statistics"""
        print("\n" + "="*70)
        print(" 📊 DATABASE STATISTICS")
        print("="*70)
        
        stats = self.db.get_statistics()
        if not stats:
            print("\n❌ Unable to retrieve statistics")
            return
        
        print(f"\n📈 Overview:")
        print(f"   Total Candles: {stats['total_rows']:,}")
        print(f"   Database Size: {stats['db_size']}")
        
        if stats.get('compression'):
            before, after, savings = stats['compression']
            print(f"\n💾 Compression:")
            print(f"   Before: {before}")
            print(f"   After: {after}")
            print(f"   Savings: {savings}%")
        
        print(f"\n📊 Data by Pair:")
        print(f"   {'Pair':<12} {'Timeframe':<10} {'Count':<15} {'Date Range'}")
        print(f"   {'-'*65}")
        
        for pair, tf, count, first, last in stats['stats']:
            date_range = f"{first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')}"
            print(f"   {pair:<12} {tf:<10} {count:<15,} {date_range}")
        
        # Check feature tables
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tablename, 
                       pg_size_pretty(pg_total_relation_size('public.'||tablename))
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE '%_features'
                ORDER BY tablename
            """)
            feature_tables = cursor.fetchall()
            
            if feature_tables:
                print(f"\n🧮 Feature Tables:")
                for table, size in feature_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"   {table:<25} {count:>10,} rows    {size}")
            
            cursor.close()
        finally:
            self.db.return_connection(conn)
        
        print()
    
    def run_feature_engineering(self):
        """Run feature engineering"""
        print("\n" + "="*70)
        print(" 🧮 FEATURE ENGINEERING")
        print("="*70)
        
        print("\nThis will calculate 60-70 technical indicators for each timeframe:")
        print("  • RSI, MACD, Bollinger Bands, ATR")
        print("  • Moving Averages (SMA, EMA)")
        print("  • Volume indicators (OBV, Volume MA)")
        print("  • Statistical features")
        print("  • And more...")
        
        print("\nOptions:")
        print("1. Process all timeframes (5min, 15min, 1hr, 4hr, 1day)")
        print("2. Process single timeframe")
        print("3. Test on limited data (100 rows)")
        print("4. Back to main menu")
        
        choice = input("\nSelect (1-4): ").strip()
        
        if choice == '4':
            return
        
        logger.info("Starting feature engineering...")
        
        try:
            subprocess.run([
                sys.executable, 
                'scripts/feature_engineering.py'
            ], input=choice + '\n', text=True, check=True)
            print("\n✅ Feature engineering completed!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Feature engineering failed: {e}")
        except KeyboardInterrupt:
            print("\n⚠️  Feature engineering interrupted by user")
    
    def auto_run(self):
        """Automated full pipeline"""
        print("\n" + "="*70)
        print(" 🚀 AUTOMATED FULL PIPELINE")
        print("="*70)
        print("\nThis will:")
        print("  1. Check database status")
        print("  2. Run backfill if needed")
        print("  3. Fill any data gaps")
        print("  4. Start real-time collection")
        
        confirm = input("\nContinue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        
        logger.info("Starting automated pipeline...")
        
        # Step 1: Check if backfill needed
        if self.db.is_database_empty():
            print("\n📥 Database empty - running backfill...")
            self.run_backfill()
        else:
            print("\n✅ Database has data - skipping backfill")
        
        # Step 2: Fill gaps
        print("\n🔍 Checking for gaps...")
        self.run_gap_filler()
        
        # Step 3: Start real-time
        print("\n📡 Starting real-time collection...")
        self.run_realtime()
    
    def run(self):
        """Main execution loop"""
        self.display_banner()
        
        # Test systems first
        if not self.test_systems():
            print("\n❌ System check failed! Please fix issues before continuing.")
            return
        
        while True:
            try:
                self.show_menu()
                choice = input("\nSelect option (1-8): ").strip()
                
                if choice == '1':
                    self.run_backfill()
                elif choice == '2':
                    self.run_gap_filler()
                elif choice == '3':
                    self.run_realtime()
                elif choice == '4':
                    self.show_statistics()
                elif choice == '5':
                    self.test_systems()
                elif choice == '6':
                    self.auto_run()
                elif choice == '7':
                    self.run_feature_engineering()
                elif choice == '8':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("\n❌ Invalid option. Please select 1-8.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")
        
        # Cleanup
        self.db.close_all_connections()


def main():
    """Entry point"""
    orchestrator = MasterOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
