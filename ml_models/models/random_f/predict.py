"""
Live Prediction Pipeline
Loads trained model and generates signals from latest database data
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/crypto_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models')

# Minimum candles needed (200 for SMA_200 + buffer)
MIN_CANDLES_1M = 3500  # ~58 hours of 1-min data for 200 15-min candles


def load_models():
    """Load trained models, scaler, features, and thresholds."""
    print("Loading models...")

    classifier = joblib.load(os.path.join(MODEL_DIR, 'direction_model.pkl'))
    regressor = joblib.load(os.path.join(MODEL_DIR, 'regression_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))

    with open(os.path.join(MODEL_DIR, 'threshold.json'), 'r') as f:
        thresholds = json.load(f)

    print(f"  Classifier: {type(classifier).__name__}")
    print(f"  Features: {len(features)}")
    print(f"  Long Threshold: {thresholds['threshold_long']}")
    print(f"  Short Threshold: {thresholds['threshold_short']}")

    return classifier, regressor, scaler, features, thresholds


def fetch_latest_data(engine, lookback_candles=MIN_CANDLES_1M):
    """Fetch latest 1-min OHLCV data from database."""
    print(f"Fetching latest {lookback_candles} candles from database...")

    query = f"""
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
        ORDER BY time DESC
        LIMIT {lookback_candles}
    """

    df = pd.read_sql(query, engine)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df = df.sort_index()  # Ascending order

    print(f"  Loaded {len(df)} rows")
    print(f"  Time range: {df.index[0]} to {df.index[-1]}")

    return df


def resample_to_15min(df):
    """Resample 1-min data to 15-min candles."""
    print("Resampling to 15-min candles...")

    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'num_trades': 'sum'
    }

    df_15m = df.resample('15T').agg(aggregation_rules)
    df_15m.dropna(inplace=True)

    print(f"  Resampled to {len(df_15m)} 15-min candles")

    return df_15m


def generate_features(df):
    """
    Generate ALL features exactly as in training.
    MUST match feature_engineering.py exactly!
    """
    print("Generating features...")

    # ==========================================================================
    # CATEGORY 1: BASIC PRICE FEATURES
    # ==========================================================================
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_range'] = df['high'] - df['low']
    df['price_range_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-10) * 100
    df['body'] = df['close'] - df['open']
    df['body_pct'] = (df['close'] - df['open']) / (df['open'] + 1e-10) * 100
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['price_acceleration'] = df['returns'].diff()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # ==========================================================================
    # CATEGORY 2: MOVING AVERAGES
    # ==========================================================================
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'distance_from_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}'] * 100

    for span in [12, 26, 50]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    df['sma_cross_5_20'] = ((df['sma_5'] > df['sma_20']).astype(int) -
                            (df['sma_5'] <= df['sma_20']).astype(int))
    df['sma_50_slope'] = df['sma_50'].diff(5)
    df['sma_200_slope'] = df['sma_200'].diff(10)

    # Higher timeframe EMAs
    df_1h = df.resample('1h').agg({'close': 'last'})
    df_1h['ema_20_1h'] = df_1h['close'].ewm(span=20, adjust=False).mean()
    df_1h['ema_50_1h'] = df_1h['close'].ewm(span=50, adjust=False).mean()
    df_1h['ema_200_1h'] = df_1h['close'].ewm(span=200, adjust=False).mean()

    df_4h = df.resample('4h').agg({'close': 'last'})
    df_4h['ema_20_4h'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema_200_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()

    df['ema_20_1h'] = df_1h['ema_20_1h'].reindex(df.index, method='ffill')
    df['ema_50_1h'] = df_1h['ema_50_1h'].reindex(df.index, method='ffill')
    df['ema_200_1h'] = df_1h['ema_200_1h'].reindex(df.index, method='ffill')
    df['ema_20_4h'] = df_4h['ema_20_4h'].reindex(df.index, method='ffill')
    df['ema_200_4h'] = df_4h['ema_200_4h'].reindex(df.index, method='ffill')

    df['dist_ema_20_1h'] = (df['close'] - df['ema_20_1h']) / df['ema_20_1h']
    df['dist_ema_200_1h'] = (df['close'] - df['ema_200_1h']) / df['ema_200_1h']
    df['dist_ema_200_4h'] = (df['close'] - df['ema_200_4h']) / df['ema_200_4h']

    # ==========================================================================
    # CATEGORY 3: VOLUME FEATURES
    # ==========================================================================
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
    df['is_volume_spike'] = (df['volume'] > df['volume'].quantile(0.95)).astype(int)
    df['volume_price_trend'] = df['volume'] * df['returns']
    df['trades_per_btc'] = df['num_trades'] / (df['volume'] + 1e-10)
    df['avg_trade_size'] = df['volume'] / (df['num_trades'] + 1e-10)
    df['volume_roc'] = df['volume'].pct_change(10)

    obv_raw = (np.sign(df['returns']) * df['volume']).fillna(0)
    df['obv_20'] = obv_raw.rolling(window=20).sum()
    df['obv_50'] = obv_raw.rolling(window=50).sum()
    df['obv_divergence'] = (df['obv_20'] - df['obv_50']) / (df['obv_50'].abs() + 1e-10)

    df['force_index'] = df['returns'] * df['volume']
    df['force_index_ema'] = df['force_index'].ewm(span=13).mean()

    ad_raw = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10) * df['volume']
    df['ad_20'] = ad_raw.rolling(window=20).sum()
    df['ad_50'] = ad_raw.rolling(window=50).sum()
    df['ad_oscillator'] = df['ad_20'] - df['ad_50']
    df['cmf_20'] = ad_raw.rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)

    # ==========================================================================
    # CATEGORY 4: VOLATILITY FEATURES
    # ==========================================================================
    for window in [10, 20, 50]:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']

    df['keltner_middle'] = df['ema_26']
    df['keltner_upper'] = df['keltner_middle'] + (2 * df['atr_14'])
    df['keltner_lower'] = df['keltner_middle'] - (2 * df['atr_14'])

    # ==========================================================================
    # CATEGORY 5: MOMENTUM INDICATORS
    # ==========================================================================
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_7'] = calculate_rsi(df['close'], 7)

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_hist_momentum'] = df['macd_histogram'].diff()
    df['macd_hist_acceleration'] = df['macd_hist_momentum'].diff()

    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-10)) * 100
    df['roc_20'] = ((df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-10)) * 100

    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)

    # Stochastic
    def calculate_stochastic(df, period=14):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(window=3).mean()
        return k, d

    df['stoch_k_14'], df['stoch_d_14'] = calculate_stochastic(df, 14)

    # Williams %R
    def calculate_williams_r(df, period=14):
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)

    df['williams_r_14'] = calculate_williams_r(df, 14)

    # MFI
    def calculate_mfi(df, period=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        return 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))

    df['mfi_14'] = calculate_mfi(df, 14)

    # CCI
    def calculate_cci(df, period=20):
        tp = df['typical_price']
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma_tp) / (0.015 * mad + 1e-10)

    df['cci_20'] = calculate_cci(df, 20)

    # ==========================================================================
    # CATEGORY 6: ADVANCED INDICATORS
    # ==========================================================================
    # ADX (FIXED with proper index alignment)
    def calculate_adx(df, period=14):
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        plus_dm_series = pd.Series(plus_dm, index=df.index)
        minus_dm_series = pd.Series(minus_dm, index=df.index)
        atr = df['tr'].rolling(window=period).mean()
        plus_di = 100 * plus_dm_series.rolling(window=period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm_series.rolling(window=period).mean() / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    df['adx_14'], df['plus_di_14'], df['minus_di_14'] = calculate_adx(df, 14)
    df['adx_trend_strength'] = np.where(df['adx_14'] > 25, 1, 0)

    # VWAP (FIXED with daily reset)
    def calculate_daily_vwap(df):
        df_temp = df.copy()
        df_temp['date'] = df_temp.index.date
        df_temp['pv'] = df_temp['typical_price'] * df_temp['volume']
        df_temp['cumsum_pv'] = df_temp.groupby('date')['pv'].cumsum()
        df_temp['cumsum_v'] = df_temp.groupby('date')['volume'].cumsum()
        return df_temp['cumsum_pv'] / (df_temp['cumsum_v'] + 1e-10)

    df['vwap'] = calculate_daily_vwap(df)
    df['vwap_distance'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10) * 100

    # Volatility Percentile
    df['vol_percentile'] = df['volatility_20'].rolling(window=100).apply(
        lambda x: (x.iloc[-1] >= x).mean() * 100 if len(x) > 0 else 50, raw=False
    )
    df['vol_regime'] = pd.cut(df['vol_percentile'], bins=[0, 25, 75, 100], labels=[0, 1, 2])
    df['vol_regime'] = df['vol_regime'].astype(float)
    df['vol_trend'] = np.sign(df['volatility_20'].diff(5))

    # Divergence Detection
    def detect_divergence(price_series, indicator_series, window=14):
        price_roll_min = price_series.rolling(window).min()
        price_roll_max = price_series.rolling(window).max()
        ind_roll_min = indicator_series.rolling(window).min()
        ind_roll_max = indicator_series.rolling(window).max()
        bearish_div = ((price_series >= price_roll_max.shift(1)) &
                       (indicator_series < ind_roll_max.shift(1))).astype(int)
        bullish_div = ((price_series <= price_roll_min.shift(1)) &
                       (indicator_series > ind_roll_min.shift(1))).astype(int)
        return bullish_div, bearish_div

    df['rsi_bullish_div'], df['rsi_bearish_div'] = detect_divergence(df['close'], df['rsi_14'], 14)
    df['macd_bullish_div'], df['macd_bearish_div'] = detect_divergence(df['close'], df['macd'], 14)

    # Returns Distribution
    df['returns_skew_20'] = df['returns'].rolling(window=20).skew()
    df['returns_kurtosis_20'] = df['returns'].rolling(window=20).kurt()

    # Normalized Momentum
    df['momentum_norm_10'] = df['momentum_10'] / (df['atr_14'] + 1e-10)
    df['momentum_norm_20'] = df['momentum_20'] / (df['atr_14'] + 1e-10)
    df['momentum_10_abs'] = np.abs(df['momentum_10'])
    df['momentum_accel'] = df['momentum_10'].diff()

    # ==========================================================================
    # CATEGORY 7: TIME-BASED FEATURES
    # ==========================================================================
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['is_us_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
    df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
    df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)

    # ==========================================================================
    # CATEGORY 8: LAG FEATURES
    # ==========================================================================
    for lag in [1, 5, 10, 30]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
    for lag in [1, 5, 10]:
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    df['rsi_lag_1'] = df['rsi_14'].shift(1)
    df['rsi_lag_5'] = df['rsi_14'].shift(5)

    # ==========================================================================
    # CATEGORY 9: PATTERN FEATURES
    # ==========================================================================
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

    def count_consecutive_streak(returns_series):
        streak = []
        current_streak = 0
        for ret in returns_series:
            if pd.isna(ret):
                streak.append(0)
            elif ret > 0:
                current_streak = current_streak + 1 if current_streak > 0 else 1
                streak.append(current_streak)
            elif ret < 0:
                current_streak = current_streak - 1 if current_streak < 0 else -1
                streak.append(current_streak)
            else:
                current_streak = 0
                streak.append(0)
        return streak

    df['consecutive_streak'] = count_consecutive_streak(df['returns'])
    df['consecutive_up'] = (df['consecutive_streak'] > 0).astype(int) * df['consecutive_streak']
    df['consecutive_down'] = (df['consecutive_streak'] < 0).astype(int) * abs(df['consecutive_streak'])

    df['gap_up'] = ((df['open'] > df['close'].shift(1)) & (df['open'] - df['close'].shift(1) > df['atr_14'] * 0.5)).astype(int)
    df['gap_down'] = ((df['open'] < df['close'].shift(1)) & (df['close'].shift(1) - df['open'] > df['atr_14'] * 0.5)).astype(int)
    df['is_doji'] = (np.abs(df['body']) / (df['price_range'] + 1e-10) < 0.1).astype(int)

    # Replace infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"  Generated {len(df.columns)} total columns")

    return df


def predict_signal(classifier, regressor, scaler, features, thresholds, df):
    """
    Generate trading signal using classifier + regression validation.
    Returns signal, confidence, and position sizing recommendation.
    """
    # Get only selected features
    X = df[features].iloc[-1:].copy()

    # Fill NaN with 0 (same as training)
    X = X.fillna(0)

    # Scale
    X_scaled = scaler.transform(X)

    # Get probabilities
    probs = classifier.predict_proba(X_scaled)[0]
    prob_neutral = probs[0]
    prob_long = probs[1]
    prob_short = probs[2]

    # Get regression prediction
    regression_pred = regressor.predict(X_scaled)[0]

    # Determine signal based on thresholds
    threshold_long = thresholds['threshold_long']
    threshold_short = thresholds['threshold_short']

    # Current price for SL/TP calculation
    current_price = df['close'].iloc[-1]
    volatility = df['volatility_20'].iloc[-1]

    # Daily trend filter (SMA 200)
    daily_df = df.resample('1D').agg({'close': 'last'})
    daily_df['sma_200'] = daily_df['close'].rolling(200).mean()
    sma_200_daily = daily_df['sma_200'].iloc[-1] if len(daily_df) >= 200 else current_price
    is_uptrend = current_price > sma_200_daily

    # Decision logic
    signal = "NEUTRAL"
    confidence = "LOW"
    position_size = 0.0
    stop_loss = None
    take_profit = None

    # LONG signal
    if prob_long >= threshold_long and is_uptrend:
        signal = "LONG"

        # Option A: Confidence scoring using regression
        if regression_pred > 0.5:  # Regression agrees with long
            confidence = "HIGH"
            # Option B: Position sizing adjustment
            position_size = 0.025  # 2.5% risk when both agree
        elif regression_pred > 0:
            confidence = "MEDIUM"
            position_size = 0.02   # 2% risk (default)
        else:
            confidence = "LOW"
            position_size = 0.015  # 1.5% risk when disagreement

        # Option C: Dynamic TP based on regression magnitude
        predicted_move = abs(regression_pred) / 100
        take_profit = current_price * (1 + max(volatility * 1.5, predicted_move * 0.8))
        stop_loss = current_price * (1 - volatility)

    # SHORT signal
    elif prob_short >= threshold_short and not is_uptrend:
        signal = "SHORT"

        if regression_pred < -0.5:  # Regression agrees with short
            confidence = "HIGH"
            position_size = 0.025
        elif regression_pred < 0:
            confidence = "MEDIUM"
            position_size = 0.02
        else:
            confidence = "LOW"
            position_size = 0.015

        predicted_move = abs(regression_pred) / 100
        take_profit = current_price * (1 - max(volatility * 1.5, predicted_move * 0.8))
        stop_loss = current_price * (1 + volatility)

    return {
        "timestamp": df.index[-1].isoformat(),
        "signal": signal,
        "classifier_probs": {
            "neutral": round(prob_neutral, 4),
            "long": round(prob_long, 4),
            "short": round(prob_short, 4)
        },
        "regression_pred": f"{regression_pred:+.2f}%",
        "confidence": confidence,
        "suggested_position": position_size,
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "take_profit": round(take_profit, 2) if take_profit else None,
        "current_price": round(current_price, 2),
        "trend": "UPTREND" if is_uptrend else "DOWNTREND",
        "volatility": round(volatility * 100, 4)
    }


def main():
    """Main execution function."""
    print("=" * 70)
    print("LIVE PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Load models
    try:
        classifier, regressor, scaler, features, thresholds = load_models()
    except FileNotFoundError as e:
        print(f"\nERROR: Model files not found!")
        print(f"  Please run feature_selection.py first to train the models.")
        print(f"  Missing: {e}")
        return None

    # 2. Connect to database
    print(f"\nConnecting to database...")
    engine = create_engine(DATABASE_URL)

    # 3. Fetch latest data
    df = fetch_latest_data(engine, MIN_CANDLES_1M)

    # 4. Resample to 15-min
    df = resample_to_15min(df)

    # 5. Generate features
    df = generate_features(df)

    # 6. Validate we have all required features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"\nWARNING: Missing features: {missing_features[:10]}...")
        print("  This may cause prediction errors.")

    # 7. Make prediction
    print("\n" + "-" * 70)
    print("GENERATING PREDICTION")
    print("-" * 70)

    result = predict_signal(classifier, regressor, scaler, features, thresholds, df)

    # 8. Output result
    print(f"\n{'='*70}")
    print("PREDICTION RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    result = main()
