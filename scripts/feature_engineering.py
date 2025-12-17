"""
feature_engineering.py - Calculate Technical Indicators
Calculates 60-70 features for each timeframe table
Uses: TA-Lib (primary), pandas-ta (backup), custom calculations
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
import warnings
from sqlalchemy import create_engine

# Try importing TA libraries (graceful fallback)
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    warnings.warn("TA-Lib not available, using pandas-ta")

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    warnings.warn("pandas-ta not available")

from scripts.database import Database

# Setup logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature Engineering for Crypto Trading
    Calculates technical indicators and features for ML models
    """
    
    def __init__(self, db: Database):
        """
        Initialize Feature Engineer
        
        Args:
            db: Database instance
        """
        self.db = db
        
        # Timeframe to table mapping
        self.timeframe_tables = {
            '5min': 'btc_5min',
            '15min': 'btc_15min',
            '1hr': 'btc_1hr',
            '4hr': 'btc_4hr',
            '1day': 'btc_1day'
        }
        
        # Feature configuration
        self.config = {
            'ma_periods': [10, 20, 50, 100, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'cci_period': 20,
            'roc_period': 10
        }
        
        logger.info("FeatureEngineer initialized")
        logger.info(f"TA-Lib available: {HAS_TALIB}")
        logger.info(f"pandas-ta available: {HAS_PANDAS_TA}")
    
    def load_timeframe_data(self, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from timeframe table
        
        Args:
            timeframe: '5min', '15min', '1hr', '4hr', '1day'
            limit: Optional row limit
        
        Returns:
            DataFrame with OHLCV data
        """
        table_name = self.timeframe_tables.get(timeframe)
        if not table_name:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        conn = self.db.get_connection()
        try:
            query = f"SELECT * FROM {table_name} ORDER BY time"
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Loading {timeframe} data from {table_name}...")
            df = pd.read_sql(query, conn)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            
            logger.info(f"✓ Loaded {len(df):,} rows for {timeframe}")
            return df
            
        finally:
            self.db.return_connection(conn)
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic price action features
        
        Features (6):
        - price_change: close - open
        - price_range: high - low
        - body_size: |close - open|
        - upper_shadow: high - max(open, close)
        - lower_shadow: min(open, close) - low
        - price_position: (close - low) / (high - low)
        """
        logger.info("Calculating basic features...")
        
        # Price changes
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = np.abs(df['close'] - df['open'])
        
        # Candle shadows
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Price position within range
        df['price_position'] = np.where(
            df['price_range'] != 0,
            (df['close'] - df['low']) / df['price_range'],
            0.5
        )
        
        logger.info("✓ Basic features calculated (6 features)")
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages
        
        Features (7):
        - SMA: 10, 20, 50, 100, 200
        - EMA: 12, 26
        """
        logger.info("Calculating moving averages...")
        
        # Simple Moving Averages
        for period in self.config['ma_periods']:
            if HAS_TALIB:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            else:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in self.config['ema_periods']:
            if HAS_TALIB:
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            else:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        logger.info("✓ Moving averages calculated (7 features)")
        return df
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators
        
        Features (11):
        - RSI (1)
        - MACD, MACD Signal, MACD Histogram (3)
        - Stochastic K, D (2)
        - CCI (1)
        - ROC (1)
        - Momentum (1)
        - Williams %R (1)
        - Ultimate Oscillator (1)
        """
        logger.info("Calculating momentum indicators...")
        
        # RSI
        if HAS_TALIB:
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.config['rsi_period'])
        else:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if HAS_TALIB:
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'],
                fastperiod=self.config['macd_fast'],
                slowperiod=self.config['macd_slow'],
                signalperiod=self.config['macd_signal']
            )
        else:
            exp1 = df['close'].ewm(span=self.config['macd_fast'], adjust=False).mean()
            exp2 = df['close'].ewm(span=self.config['macd_slow'], adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.config['macd_signal'], adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        if HAS_TALIB:
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=self.config['stoch_k'],
                slowk_period=3,
                slowd_period=self.config['stoch_d']
            )
        else:
            low_min = df['low'].rolling(window=self.config['stoch_k']).min()
            high_max = df['high'].rolling(window=self.config['stoch_k']).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=self.config['stoch_d']).mean()
        
        # CCI (Commodity Channel Index)
        if HAS_TALIB:
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=self.config['cci_period'])
        else:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(window=self.config['cci_period']).mean()
            mad = tp.rolling(window=self.config['cci_period']).apply(lambda x: np.abs(x - x.mean()).mean())
            df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        # ROC (Rate of Change)
        if HAS_TALIB:
            df['roc'] = talib.ROC(df['close'], timeperiod=self.config['roc_period'])
        else:
            df['roc'] = df['close'].pct_change(periods=self.config['roc_period']) * 100
        
        # Momentum
        if HAS_TALIB:
            df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        else:
            df['momentum'] = df['close'].diff(10)
        
        # Williams %R
        if HAS_TALIB:
            df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            high_max = df['high'].rolling(window=14).max()
            low_min = df['low'].rolling(window=14).min()
            df['willr'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # Ultimate Oscillator
        if HAS_TALIB:
            df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
        else:
            # Simplified version
            df['ultosc'] = np.nan  # Skip if no TA-Lib
        
        logger.info("✓ Momentum indicators calculated (11 features)")
        return df
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators
        
        Features (8):
        - Bollinger Bands: upper, middle, lower, width (4)
        - ATR (1)
        - Standard Deviation: 10, 20, 50 periods (3)
        """
        logger.info("Calculating volatility indicators...")
        
        # Bollinger Bands
        if HAS_TALIB:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'],
                timeperiod=self.config['bb_period'],
                nbdevup=self.config['bb_std'],
                nbdevdn=self.config['bb_std']
            )
        else:
            sma = df['close'].rolling(window=self.config['bb_period']).mean()
            std = df['close'].rolling(window=self.config['bb_period']).std()
            df['bb_upper'] = sma + (std * self.config['bb_std'])
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * self.config['bb_std'])
        
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # ATR (Average True Range)
        if HAS_TALIB:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.config['atr_period'])
        else:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=self.config['atr_period']).mean()
        
        # Standard Deviations
        for period in [10, 20, 50]:
            df[f'std_{period}'] = df['close'].rolling(window=period).std()
        
        logger.info("✓ Volatility indicators calculated (8 features)")
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume indicators
        
        Features (5):
        - Volume MA (1)
        - Volume Ratio (1)
        - OBV (1)
        - Volume ROC (1)
        - Volume Momentum (1)
        """
        logger.info("Calculating volume indicators...")
        
        # Volume Moving Average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Volume Ratio
        df['volume_ratio'] = np.where(
            df['volume_ma'] != 0,
            df['volume'] / df['volume_ma'],
            1
        )
        
        # OBV (On Balance Volume)
        if HAS_TALIB:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        else:
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume ROC
        df['volume_roc'] = df['volume'].pct_change(periods=10) * 100
        
        # Volume Momentum
        df['volume_momentum'] = df['volume'].diff(10)
        
        logger.info("✓ Volume indicators calculated (5 features)")
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns over different periods
        
        Features (7):
        - Returns: 1, 5, 10, 30, 60 periods (5)
        - Log returns: 1, 5 periods (2)
        """
        logger.info("Calculating returns...")
        
        # Simple returns
        for period in [1, 5, 10, 30, 60]:
            df[f'return_{period}'] = df['close'].pct_change(periods=period) * 100
        
        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        
        logger.info("✓ Returns calculated (7 features)")
        return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical features
        
        Features (10):
        - Rolling min/max: 10, 20, 50 periods (6)
        - Z-score: 20 periods (1)
        - Percentile rank: 50 periods (1)
        - Skewness: 20 periods (1)
        - Kurtosis: 20 periods (1)
        """
        logger.info("Calculating statistical features...")
        
        # Rolling min/max
        for period in [10, 20, 50]:
            df[f'rolling_min_{period}'] = df['close'].rolling(window=period).min()
            df[f'rolling_max_{period}'] = df['close'].rolling(window=period).max()
        
        # Z-score
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - rolling_mean) / rolling_std
        
        # Percentile rank
        df['percentile_rank'] = df['close'].rolling(window=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        
        # Skewness
        df['skewness'] = df['close'].rolling(window=20).skew()
        
        # Kurtosis
        df['kurtosis'] = df['close'].rolling(window=20).kurt()
        
        logger.info("✓ Statistical features calculated (10 features)")
        return df
    
    def calculate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate lag features
        
        Features (9):
        - Close lag: 1, 5, 10, 30 periods (4)
        - Volume lag: 1, 5, 10 periods (3)
        - RSI lag: 1, 5 periods (2)
        """
        logger.info("Calculating lag features...")
        
        # Price lags
        for period in [1, 5, 10, 30]:
            df[f'close_lag_{period}'] = df['close'].shift(period)
        
        # Volume lags
        for period in [1, 5, 10]:
            df[f'volume_lag_{period}'] = df['volume'].shift(period)
        
        # RSI lags (if RSI exists)
        if 'rsi' in df.columns:
            df['rsi_lag_1'] = df['rsi'].shift(1)
            df['rsi_lag_5'] = df['rsi'].shift(5)
        
        logger.info("✓ Lag features calculated (9 features)")
        return df
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features in correct order
        
        Total features: ~70
        """
        logger.info("="*70)
        logger.info("CALCULATING ALL FEATURES")
        logger.info("="*70)
        
        # Calculate features in order
        df = self.calculate_basic_features(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_momentum_indicators(df)
        df = self.calculate_volatility_indicators(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_returns(df)
        df = self.calculate_statistical_features(df)
        df = self.calculate_lag_features(df)
        
        # Count features
        feature_cols = [col for col in df.columns if col not in ['time', 'pair', 'open', 'high', 'low', 'close', 'volume']]
        logger.info(f"✓ Total features calculated: {len(feature_cols)}")
        logger.info("="*70)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values (NaN) from rolling calculations
        
        Strategy: Remove first N rows with NaN (caused by rolling windows)
        """
        logger.info("Handling missing values...")
        
        initial_rows = len(df)
        
        # Find first row without NaN
        first_valid_idx = df.dropna().index[0] if len(df.dropna()) > 0 else 0
        
        # Remove rows with NaN
        df = df.loc[first_valid_idx:].reset_index(drop=True)
        
        removed_rows = initial_rows - len(df)
        logger.info(f"✓ Removed {removed_rows} rows with NaN (kept {len(df):,} rows)")
        
        return df
    
    def save_features_to_table(self, df: pd.DataFrame, timeframe: str):
        """
        Save feature-engineered data to new table
        
        Args:
            df: DataFrame with features
            timeframe: '5min', '15min', etc.
        """
        from sqlalchemy import create_engine
        
        table_name = self.timeframe_tables[timeframe] + '_features'
        
        logger.info(f"Saving features to {table_name}...")
        
        # Get SQLAlchemy engine using database config
        db_config = self.db.config
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        try:
            # Save DataFrame to PostgreSQL
            df.to_sql(
                table_name,
                engine,
                if_exists='replace',
                index=False,
                chunksize=5000
            )
            
            # Create time index
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {table_name}(time)")
            conn.commit()
            cursor.close()
            self.db.return_connection(conn)
            
            logger.info(f"✓ Saved {len(df):,} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"❌ Error saving to {table_name}: {e}")
            raise
        finally:
            engine.dispose()
        
    def process_timeframe(self, timeframe: str, save_to_db: bool = True) -> pd.DataFrame:
        """
        Complete processing for one timeframe
        
        Args:
            timeframe: '5min', '15min', '1hr', '4hr', '1day'
            save_to_db: Whether to save to database
        
        Returns:
            DataFrame with all features
        """
        logger.info("="*70)
        logger.info(f"PROCESSING TIMEFRAME: {timeframe}")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Load data
        df = self.load_timeframe_data(timeframe)
        
        # Calculate features
        df = self.calculate_all_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Save to database
        if save_to_db:
            self.save_features_to_table(df, timeframe)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ {timeframe} processing completed in {elapsed:.2f}s")
        logger.info("="*70)
        
        return df
    
    def process_all_timeframes(self):
        """
        Process all timeframes
        """
        logger.info("\n" + "="*70)
        logger.info("FEATURE ENGINEERING - ALL TIMEFRAMES")
        logger.info("="*70 + "\n")
        
        start_time = datetime.now()
        results = {}
        
        for timeframe in ['5min', '15min', '1hr', '4hr', '1day']:
            try:
                df = self.process_timeframe(timeframe, save_to_db=True)
                results[timeframe] = {
                    'status': 'success',
                    'rows': len(df),
                    'features': len([c for c in df.columns if c not in ['time', 'pair', 'open', 'high', 'low', 'close', 'volume']])
                }
            except Exception as e:
                logger.error(f"❌ Error processing {timeframe}: {e}")
                results[timeframe] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "="*70)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*70)
        
        for tf, result in results.items():
            if result['status'] == 'success':
                logger.info(f"✓ {tf}: {result['rows']:,} rows, {result['features']} features")
            else:
                logger.info(f"❌ {tf}: {result['error']}")
        
        logger.info(f"\nTotal time: {elapsed:.2f}s")
        logger.info("="*70 + "\n")
        
        return results


# CLI Interface
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Setup logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/feature_engineering.log'),
            logging.StreamHandler()
        ]
    )
    
    print("\n" + "="*70)
    print(" 🧮 FEATURE ENGINEERING")
    print("="*70)
    
    # Initialize
    db = Database()
    
    if not db.test_connection():
        print("\n❌ Database connection failed!")
        sys.exit(1)
    
    print("\n✅ Database connected")
    
    # Create feature engineer
    engineer = FeatureEngineer(db)
    
    # Menu
    print("\nOptions:")
    print("1. Process single timeframe")
    print("2. Process all timeframes")
    print("3. Test on limited data (100 rows)")
    
    choice = input("\nSelect (1-3): ").strip()
    
    if choice == '1':
        print("\nTimeframes: 5min, 15min, 1hr, 4hr, 1day")
        tf = input("Select timeframe: ").strip()
        
        if tf not in ['5min', '15min', '1hr', '4hr', '1day']:
            print("❌ Invalid timeframe")
            sys.exit(1)
        
        print(f"\nProcessing {tf}...")
        engineer.process_timeframe(tf, save_to_db=True)
        print("\n✅ Done!")
    
    elif choice == '2':
        confirm = input("\n⚠️  This will process all timeframes. Continue? (y/n): ")
        if confirm.lower() == 'y':
            engineer.process_all_timeframes()
            print("\n✅ All timeframes processed!")
    
    elif choice == '3':
        print("\nTest mode: Processing first 100 rows of 5min data...")
        df = engineer.load_timeframe_data('5min', limit=100)
        df = engineer.calculate_all_features(df)
        print(f"\n✅ Test complete!")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nSample features:")
        print(df[['time', 'close', 'rsi', 'macd', 'bb_upper']].tail())
    
    else:
        print("❌ Invalid choice")
    
    db.close_all_connections()
    print()
