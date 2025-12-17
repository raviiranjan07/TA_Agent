"""
database.py - PostgreSQL/TimescaleDB Connection Module
Handles all database operations for crypto data pipeline
"""

import os
import logging
from datetime import datetime
from psycopg2 import pool
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from psycopg2.extras import execute_values

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

    def get_connection(self):
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get connection: {e}")
            raise

    def return_connection(self, conn):
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
        """Get timestamp of last candle"""
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
            return result
        except Exception as e:
            logger.error(f"Failed to get last candle time: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn)

    def insert_candles(self, candles):
        if not candles:
            logger.warning("No candles to insert")
            return 0

        query = """
            INSERT INTO ohlcv_data (
                time, pair, timeframe, open, high, low, close,
                volume, quote_volume, num_trades, taker_buy_base, taker_buy_quote
            ) VALUES %s
            ON CONFLICT (pair, timeframe, time) DO NOTHING
        """

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            execute_values(cursor, query, candles, page_size=1000)
            inserted = cursor.rowcount
            conn.commit()
            logger.info(f"Inserted {inserted} candles ({len(candles) - inserted} duplicates skipped)")
            return inserted
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to insert candles: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)

    def get_data_gaps(self, pair, timeframe, start_time, end_time):
        """Find gaps in OHLCV data"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = """
                WITH ts AS (
                    SELECT time FROM ohlcv_data
                    WHERE pair=%s AND timeframe=%s AND time BETWEEN %s AND %s
                    ORDER BY time
                ),
                gaps AS (
                    SELECT time AS gap_start,
                           LEAD(time) OVER (ORDER BY time) AS gap_end
                    FROM ts
                )
                SELECT gap_start, gap_end
                FROM gaps
                WHERE gap_end - gap_start > INTERVAL '1 minute'
                ORDER BY gap_start;
            """
            cursor.execute(query, (pair, timeframe, start_time, end_time))
            gaps = cursor.fetchall()
            cursor.close()
            logger.info(f"Found {len(gaps)} gaps for {pair} {timeframe}")
            return gaps
        except Exception as e:
            logger.error(f"Failed to get data gaps: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)

    def get_statistics(self):
        """Return DB stats"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ohlcv_data;")
            total_rows = cursor.fetchone()[0]

            cursor.execute("""
                SELECT pair, timeframe, COUNT(*) as count,
                       MIN(time) as first_candle, MAX(time) as last_candle
                FROM ohlcv_data
                GROUP BY pair, timeframe
                ORDER BY pair, timeframe;
            """)
            stats = cursor.fetchall()

            cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('ohlcv_data'));")
            db_size = cursor.fetchone()[0]

            cursor.close()
            return {"total_rows": total_rows, "db_size": db_size, "stats": stats}
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn)

    def close_all_connections(self):
        """Close all DB connections"""
        try:
            self.connection_pool.closeall()
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Failed to close connections: {e}")


# Quick test when run directly
if __name__ == "__main__":
    db = Database()
    if db.test_connection():
        print("✅ Database connection successful!")
        print(f"Database empty: {db.is_database_empty()}")
        stats = db.get_statistics()
        if stats:
            print(f"Total rows: {stats['total_rows']}")
            print(f"Database size: {stats['db_size']}")
    db.close_all_connections()
