"""
database.py - PostgreSQL/TimescaleDB Connection Module
Handles all database operations for crypto data pipeline
"""

import os
import logging
from datetime import datetime
from psycopg2 import pool
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Database:
    """Database connection and operations manager"""

    def __init__(self):
        """Initialize database connection pool"""
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                1, 20,
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'crypto_data'),
                user=os.getenv('DB_USER', 'raviranjan'),
                password=os.getenv('DB_PASSWORD', '')
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @property
    def config(self):
        """Expose database configuration for SQLAlchemy"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'crypto_data'),
            'user': os.getenv('DB_USER', 'raviranjan'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
    def get_connection(self):
        """Get connection from pool"""
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get connection: {e}")
            raise

    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)

    def test_connection(self):
        """Test database connection"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            logger.info(f"Database connected: {version[0]}")

            # Check TimescaleDB
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
            ts_version = cursor.fetchone()
            if ts_version:
                logger.info(f"TimescaleDB version: {ts_version[0]}")
            else:
                logger.warning("TimescaleDB extension not found!")

            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
        finally:
            if conn:
                self.return_connection(conn)

    def is_database_empty(self):
        """Check if ohlcv_data table is empty"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ohlcv_data;")
            count = cursor.fetchone()[0]
            cursor.close()
            logger.info(f"Database has {count} rows")
            return count == 0
        except Exception as e:
            logger.error(f"Failed to check if database is empty: {e}")
            return True
        finally:
            if conn:
                self.return_connection(conn)

    def get_last_candle_time(self, pair, timeframe):
        """Get timestamp of last candle for a pair/timeframe"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = """
                SELECT MAX(time) FROM ohlcv_data
                WHERE pair = %s AND timeframe = %s;
            """
            cursor.execute(query, (pair, timeframe))
            result = cursor.fetchone()[0]
            cursor.close()
            
            if result:
                logger.debug(f"Last candle for {pair} {timeframe}: {result}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to get last candle time: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn)

    def insert_candles(self, candles):
        """
        Insert candles with duplicate handling
        
        Args:
            candles: List of tuples (time, pair, timeframe, open, high, low, close, 
                     volume, quote_volume, num_trades, taker_buy_base, taker_buy_quote)
        
        Returns:
            Number of rows inserted (excluding duplicates)
        """
        if not candles:
            logger.warning("No candles to insert")
            return 0

        # FIXED: Match PRIMARY KEY order (time, pair, timeframe)
        query = """
            INSERT INTO ohlcv_data (
                time, pair, timeframe, open, high, low, close,
                volume, quote_volume, num_trades, taker_buy_base, taker_buy_quote
            ) VALUES %s
            ON CONFLICT (time, pair, timeframe) DO NOTHING
        """

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Use execute_values for bulk insert
            execute_values(cursor, query, candles, page_size=1000)
            inserted = cursor.rowcount
            conn.commit()
            
            duplicates = len(candles) - inserted
            if duplicates > 0:
                logger.debug(f"Inserted {inserted} candles ({duplicates} duplicates skipped)")
            else:
                logger.info(f"Inserted {inserted} candles")
            
            cursor.close()
            return inserted
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to insert candles: {e}")
            logger.error(f"First candle sample: {candles[0] if candles else 'None'}")
            raise
        finally:
            if conn:
                self.return_connection(conn)

    def get_data_gaps(self, pair, timeframe, start_time, end_time, min_gap_minutes=5):
        """
        Find gaps in OHLCV data
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
            start_time: Start datetime
            end_time: End datetime
            min_gap_minutes: Minimum gap size to report (default 5 minutes)
        
        Returns:
            List of (gap_start, gap_end) tuples
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                WITH ts AS (
                    SELECT time FROM ohlcv_data
                    WHERE pair = %s 
                      AND timeframe = %s 
                      AND time BETWEEN %s AND %s
                    ORDER BY time
                ),
                gaps AS (
                    SELECT 
                        time AS gap_start,
                        LEAD(time) OVER (ORDER BY time) AS gap_end
                    FROM ts
                )
                SELECT gap_start, gap_end
                FROM gaps
                WHERE gap_end - gap_start > INTERVAL '%s minutes'
                ORDER BY gap_start;
            """
            
            cursor.execute(query, (pair, timeframe, start_time, end_time, min_gap_minutes))
            gaps = cursor.fetchall()
            cursor.close()
            
            if gaps:
                logger.info(f"Found {len(gaps)} gaps for {pair} {timeframe}")
            else:
                logger.info(f"No gaps found for {pair} {timeframe}")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to get data gaps: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)

    def get_statistics(self):
        """Get database statistics"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Total rows
            cursor.execute("SELECT COUNT(*) FROM ohlcv_data;")
            total_rows = cursor.fetchone()[0]

            # Per pair/timeframe stats
            cursor.execute("""
                SELECT 
                    pair, 
                    timeframe, 
                    COUNT(*) as count,
                    MIN(time) as first_candle, 
                    MAX(time) as last_candle
                FROM ohlcv_data
                GROUP BY pair, timeframe
                ORDER BY pair, timeframe;
            """)
            stats = cursor.fetchall()

            # Database size
            cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('ohlcv_data'));")
            db_size = cursor.fetchone()[0]
            
            # Compression info (TimescaleDB)
            try:
                cursor.execute("""
                    SELECT 
                        pg_size_pretty(before_compression_total_bytes) as before,
                        pg_size_pretty(after_compression_total_bytes) as after,
                        ROUND(100 - (after_compression_total_bytes::numeric / 
                              before_compression_total_bytes::numeric * 100), 2) as savings_pct
                    FROM timescaledb_information.compression_settings cs
                    JOIN timescaledb_information.compressed_chunk_stats ccs 
                      ON cs.hypertable_name = ccs.hypertable_name
                    WHERE cs.hypertable_name = 'ohlcv_data'
                    LIMIT 1;
                """)
                compression = cursor.fetchone()
            except:
                compression = None

            cursor.close()
            
            return {
                "total_rows": total_rows, 
                "db_size": db_size, 
                "stats": stats,
                "compression": compression
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn)

    def get_candle_count_by_date(self, pair, timeframe, start_date, end_date):
        """Get count of candles for a date range"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT COUNT(*) FROM ohlcv_data
                WHERE pair = %s 
                  AND timeframe = %s 
                  AND time BETWEEN %s AND %s;
            """
            
            cursor.execute(query, (pair, timeframe, start_date, end_date))
            count = cursor.fetchone()[0]
            cursor.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to get candle count: {e}")
            return 0
        finally:
            if conn:
                self.return_connection(conn)

    def close_all_connections(self):
        """Close all database connections"""
        try:
            self.connection_pool.closeall()
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Failed to close connections: {e}")


# Quick test when run directly
if __name__ == "__main__":
    import sys
    
    print("Testing database connection...")
    
    db = Database()
    
    if db.test_connection():
        print("✅ Database connection successful!")
        
        is_empty = db.is_database_empty()
        print(f"Database empty: {is_empty}")
        
        if not is_empty:
            stats = db.get_statistics()
            if stats:
                print(f"\nStatistics:")
                print(f"  Total rows: {stats['total_rows']:,}")
                print(f"  Database size: {stats['db_size']}")
                
                if stats['compression']:
                    before, after, savings = stats['compression']
                    print(f"\nCompression:")
                    print(f"  Before: {before}")
                    print(f"  After: {after}")
                    print(f"  Savings: {savings}%")
                
                print(f"\nData by pair:")
                for pair, tf, count, first, last in stats['stats']:
                    print(f"  {pair} {tf}: {count:,} candles ({first} to {last})")
    else:
        print("❌ Database connection failed!")
        sys.exit(1)
    
    db.close_all_connections()