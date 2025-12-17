"""
Step 3: Enhanced Feature Engineering
Create diverse features for better prediction
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 3: ENHANCED FEATURE ENGINEERING")
print("="*70)

# Connect and load data
engine = create_engine('postgresql://raviranjan@localhost/crypto_data')

print("\n📥 Loading data...")
query = """
    SELECT 
        time,
        open,
        high,
        low,
        close,
        volume,
        num_trades
    FROM ohlcv_data
    WHERE pair = 'BTCUSDT'
      AND time >= '2021-01-01'
      AND time < '2025-12-15'
    ORDER BY time
"""

df = pd.read_sql(query, engine)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

print(f"✅ Loaded {len(df):,} rows of 1-minute data")

#=============================================================================
# [INSERT THIS BLOCK HERE] RESAMPLE TO 5-MINUTE TIMEFRAME
# =============================================================================
print("\n🔄 Resampling to 5-Minute Candles...")

# Define how to aggregate each column
aggregation_rules = {
    'open': 'first',    # First price of the 5 mins
    'high': 'max',      # Highest price during the 5 mins
    'low': 'min',       # Lowest price during the 5 mins
    'close': 'last',    # Last price of the 5 mins
    'volume': 'sum',    # Total volume
    'num_trades': 'sum' # Total trades
}

# Resample and drop incomplete rows
df = df.resample('15T').agg(aggregation_rules)
df.dropna(inplace=True)

print(f"✅ Resampled to {len(df):,} rows of 5-minute data\n")
# =============================================================================

# =============================================================================
# CATEGORY 1: BASIC PRICE FEATURES
# =============================================================================
print("="*70)
print("CATEGORY 1: Basic Price Features")
print("="*70)

# Returns
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# Price range
df['price_range'] = df['high'] - df['low']
df['price_range_pct'] = (df['high'] - df['low']) / df['close'] * 100

# Body size
df['body'] = df['close'] - df['open']
df['body_pct'] = (df['close'] - df['open']) / df['open'] * 100

# Shadows
df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

# Price position
df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

# NEW: Price acceleration
df['price_acceleration'] = df['returns'].diff()

# NEW: Typical Price (used in many indicators)
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

print("✅ Basic price features (10 features)")

# =============================================================================
# CATEGORY 2: MOVING AVERAGES (TREND FEATURES)
# =============================================================================
print("\n" + "="*70)
print("CATEGORY 2: Moving Averages (Trend)")
print("="*70)

# Simple Moving Averages
for window in [5, 10, 20, 50, 100, 200]:
    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
    df[f'distance_from_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}'] * 100

# Exponential Moving Averages - FIX: Add 12 for MACD!
for span in [12, 26, 50]:  # Added 12!
    df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

# Moving average crossovers
df['sma_cross_5_20'] = ((df['sma_5'] > df['sma_20']).astype(int) - 
                         (df['sma_5'] <= df['sma_20']).astype(int))

# NEW: Trend strength (slope of moving average)
df['sma_50_slope'] = df['sma_50'].diff(5)
df['sma_200_slope'] = df['sma_200'].diff(10)

print("✅ Moving averages (20 features)")

print("   Calculing 15m Indicators...")
df['sma_20'] = df['close'].rolling(window=20).mean()
df['sma_50'] = df['close'].rolling(window=50).mean()
df['sma_200'] = df['close'].rolling(window=200).mean()

# --- NEW: ADD HIGHER TIMEFRAME CONTEXT (THE X-RAY VISION) ---
print("   Adding 1-Hour and 4-Hour Context...")

# 1. Resample to 1-Hour and calculate EMA
df_1h = df.resample('1h').agg({'close': 'last'})
df_1h['ema_20_1h'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['ema_50_1h'] = df_1h['close'].ewm(span=50, adjust=False).mean()
df_1h['ema_200_1h'] = df_1h['close'].ewm(span=200, adjust=False).mean()

# 2. Resample to 4-Hour and calculate EMA
df_4h = df.resample('4h').agg({'close': 'last'})
df_4h['ema_20_4h'] = df_4h['close'].ewm(span=20, adjust=False).mean()
df_4h['ema_200_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()

# 3. Map these back to the 15-minute Data
# (We use reindex + ffill so every 15m candle knows the current 1H/4H trend)
df['ema_20_1h'] = df_1h['ema_20_1h'].reindex(df.index, method='ffill')
df['ema_50_1h'] = df_1h['ema_50_1h'].reindex(df.index, method='ffill')
df['ema_200_1h'] = df_1h['ema_200_1h'].reindex(df.index, method='ffill')

df['ema_20_4h'] = df_4h['ema_20_4h'].reindex(df.index, method='ffill')
df['ema_200_4h'] = df_4h['ema_200_4h'].reindex(df.index, method='ffill')

# 4. Create "Distance" Features (This is what the Model learns!)
# "How far is price from the 4H trend?"
df['dist_ema_20_1h'] = (df['close'] - df['ema_20_1h']) / df['ema_20_1h']
df['dist_ema_200_1h'] = (df['close'] - df['ema_200_1h']) / df['ema_200_1h']
df['dist_ema_200_4h'] = (df['close'] - df['ema_200_4h']) / df['ema_200_4h']

print("   ✅ MTF Context Added (1H & 4H EMAs)")

# =============================================================================
# CATEGORY 3: VOLUME FEATURES (ENHANCED)
# =============================================================================
print("\n" + "="*70)
print("CATEGORY 3: Enhanced Volume Features")
print("="*70)

# Basic volume features
df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

# Volume spikes
df['is_volume_spike'] = (df['volume'] > df['volume'].quantile(0.95)).astype(int)

# Price-volume relationship
df['volume_price_trend'] = df['volume'] * df['returns']

# Trade features
df['trades_per_btc'] = df['num_trades'] / (df['volume'] + 1e-10)
df['avg_trade_size'] = df['volume'] / (df['num_trades'] + 1e-10)

# NEW: Volume Rate of Change
df['volume_roc'] = df['volume'].pct_change(10)

# NEW: On-Balance Volume (OBV)
df['obv'] = (np.sign(df['returns']) * df['volume']).fillna(0).cumsum()
df['obv_sma'] = df['obv'].rolling(window=20).mean()
df['obv_divergence'] = (df['obv'] - df['obv_sma']) / (df['obv_sma'] + 1e-10)

# NEW: Force Index
df['force_index'] = df['returns'] * df['volume']
df['force_index_ema'] = df['force_index'].ewm(span=13).mean()

# NEW: Accumulation/Distribution Line
df['ad_line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10) * df['volume']
df['ad_line'] = df['ad_line'].fillna(0).cumsum()

print("✅ Volume features (15 features)")

# =============================================================================
# CATEGORY 4: VOLATILITY FEATURES (ENHANCED)
# =============================================================================
print("\n" + "="*70)
print("CATEGORY 4: Enhanced Volatility Features")
print("="*70)

# Rolling standard deviation
for window in [10, 20, 50]:
    df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

# NEW: Bollinger %B (critical feature!)
df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

# Average True Range (ATR)
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr_14'] = df['tr'].rolling(window=14).mean()

# NEW: Normalized ATR (ATR relative to price)
df['atr_ratio'] = df['atr_14'] / df['close']

# NEW: Keltner Channels
df['keltner_middle'] = df['ema_26']
df['keltner_upper'] = df['keltner_middle'] + (2 * df['atr_14'])
df['keltner_lower'] = df['keltner_middle'] - (2 * df['atr_14'])

print("✅ Volatility features (15 features)")

# =============================================================================
# CATEGORY 5: MOMENTUM INDICATORS (ENHANCED)
# =============================================================================
print("\n" + "="*70)
print("CATEGORY 5: Enhanced Momentum Indicators")
print("="*70)

# RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi_14'] = calculate_rsi(df['close'], 14)
df['rsi_7'] = calculate_rsi(df['close'], 7)  # Faster RSI

# MACD (FIXED!)
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_histogram'] = df['macd'] - df['macd_signal']

# Rate of Change
df['roc_10'] = ((df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-10)) * 100
df['roc_20'] = ((df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-10)) * 100

# Momentum
df['momentum_10'] = df['close'] - df['close'].shift(10)
df['momentum_20'] = df['close'] - df['close'].shift(20)

# NEW: Stochastic Oscillator
def calculate_stochastic(df, period=14):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(window=3).mean()
    return k, d

df['stoch_k_14'], df['stoch_d_14'] = calculate_stochastic(df, 14)

# NEW: Williams %R
def calculate_williams_r(df, period=14):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    wr = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
    return wr

df['williams_r_14'] = calculate_williams_r(df, 14)

# NEW: Money Flow Index (MFI)
def calculate_mfi(df, period=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
    return mfi

df['mfi_14'] = calculate_mfi(df, 14)

# NEW: Commodity Channel Index (CCI)
def calculate_cci(df, period=20):
    tp = df['typical_price']
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    return cci

df['cci_20'] = calculate_cci(df, 20)

print("✅ Momentum indicators (18 features)")

# =============================================================================
# CATEGORY 6: TIME-BASED FEATURES
# =============================================================================
print("\n" + "="*70)
print("CATEGORY 6: Time-Based Features")
print("="*70)

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Market sessions
df['is_us_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)

print("✅ Time-based features (11 features)")

# =============================================================================
# CATEGORY 7: LAG FEATURES
# =============================================================================
print("\n" + "="*70)
print("CATEGORY 7: Lag Features")
print("="*70)

# Lagged returns
for lag in [1, 5, 10, 30]:
    df[f'return_lag_{lag}'] = df['returns'].shift(lag)

# Lagged volumes
for lag in [1, 5, 10]:
    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

# NEW: Lagged RSI
df['rsi_lag_1'] = df['rsi_14'].shift(1)
df['rsi_lag_5'] = df['rsi_14'].shift(5)

print("✅ Lag features (9 features)")

# =============================================================================
# CATEGORY 8: PATTERN FEATURES (NEW!)
# =============================================================================
print("\n" + "="*70)
print("CATEGORY 8: Pattern Features (NEW!)")
print("="*70)

# Higher highs / Lower lows
df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

# Consecutive up/down days
df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
for i in range(2, 6):
    df['consecutive_up'] = df['consecutive_up'] * (df['close'].shift(i-1) > df['close'].shift(i)).astype(int) + df['consecutive_up']

# Gap detection
df['gap_up'] = ((df['open'] > df['close'].shift(1)) & (df['open'] - df['close'].shift(1) > df['atr_14'] * 0.5)).astype(int)
df['gap_down'] = ((df['open'] < df['close'].shift(1)) & (df['close'].shift(1) - df['open'] > df['atr_14'] * 0.5)).astype(int)

# Doji pattern (small body relative to range)
df['is_doji'] = (np.abs(df['body']) / (df['price_range'] + 1e-10) < 0.1).astype(int)

print("✅ Pattern features (6 features)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ENHANCED FEATURE ENGINEERING COMPLETE!")
print("="*70)

original_cols = ['open', 'high', 'low', 'close', 'volume', 'num_trades']
feature_cols = [col for col in df.columns if col not in original_cols]

print(f"\n📊 Original columns: {len(original_cols)}")
print(f"📊 New features created: {len(feature_cols)}")
print(f"📊 Total columns: {len(df.columns)}")

# Feature breakdown
categories = {
    'Basic Price': 10,
    'Moving Averages': 20,
    'Volume': 15,
    'Volatility': 15,
    'Momentum': 18,
    'Time-Based': 11,
    'Lag': 9,
    'Pattern': 6
}

print("\n📋 Feature Breakdown:")
for cat, count in categories.items():
    print(f"  • {cat:20s}: {count:3d} features")
print(f"  {'TOTAL':20s}: {sum(categories.values()):3d} features")

# Check for missing values
print(f"\n⚠️  Missing values per column (top 10):")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).head(10))

# Save
print(f"\n💾 Saving engineered features...")
df.to_csv('data/btc_with_features.csv')
print("✅ Saved to: data/btc_with_features.csv")

print("\n" + "="*70)
print("✅ READY FOR MACHINE LEARNING WITH 104 DIVERSE FEATURES!")
print("="*70)