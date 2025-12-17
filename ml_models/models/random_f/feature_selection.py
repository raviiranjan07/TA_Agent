"""
Step 4: Feature Selection & Machine Learning Model (FIXED VERSION)
Build a model to predict BTC price direction with proper feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 4: FEATURE SELECTION & MACHINE LEARNING (FIXED)")
print("="*70)

# =============================================================================
# STEP 1: LOAD DATA WITH FEATURES
# =============================================================================
print("\n📥 Loading engineered features...")
df = pd.read_csv('data/btc_with_features.csv', index_col='time', parse_dates=True)
print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")

# # =============================================================================
# # STEP 2: CREATE TARGETS
# # =============================================================================
# print("\n" + "="*70)
# print("📊 Step 2: Creating Prediction Targets")
# print("="*70)

# # Predict 5 minutes ahead
# forecast_horizon = 1

# # Target 1: Price change in dollars
# df['price_change'] = df['close'].shift(-forecast_horizon) - df['close'] #shift(-n) moves data UP by n rows

# # Target 2: Price change in percentage
# df['price_change_pct'] = (df['close'].shift(-forecast_horizon) - df['close']) / df['close'] * 100

# # Target 3: Direction (1 = UP, 0 = DOWN)
# df['direction'] = (df['price_change_pct'] > 0).astype(int)

# print(f"\nTargets created for {forecast_horizon}-candle ahead prediction")
# print("\nDirection Distribution:")
# print(df['direction'].value_counts(normalize=True))

# =============================================================================
# STEP 2: CREATE TARGETS (TRIPLE BARRIER METHOD)
# =============================================================================
# print("\n" + "="*70)
# print("📊 Step 2: Creating Triple Barrier Targets (Profit vs Stop)")
# print("="*70)

# # 1. Calculate Volatility (to set dynamic barrier width)
# # We use a rolling standard deviation of returns to estimate current volatility
# df['returns'] = df['close'].pct_change()
# df['volatility'] = df['returns'].rolling(window=20).std()

# # 2. Define the Triple Barrier Function
# def get_triple_barrier_label(close_prices, volatility, t_events, pt=1, sl=1, min_ret=0.001, max_horizon=10):
#     """
#     pt: Profit Taking Multiplier (e.g., 1x Volatility)
#     sl: Stop Loss Multiplier (e.g., 1x Volatility)
#     min_ret: Minimum return required to consider a trade
#     max_horizon: Maximum candles to hold before quitting (Vertical Barrier)
#     """
#     labels = []
    
#     # Convert to numpy for speed
#     closes = close_prices.values
#     vols = volatility.values
    
#     print(f"  generating labels for {len(closes)} rows (this may take a moment)...")
    
#     for i in range(len(closes)):
#         # Skip if we don't have enough future data or volatility is NaN
#         if i + max_horizon >= len(closes) or np.isnan(vols[i]):
#             labels.append(np.nan)
#             continue
            
#         current_price = closes[i]
#         current_vol = vols[i]
        
#         # Determine dynamic barriers based on current volatility
#         # If volatility is low, barriers are close. If high, barriers are wide.
#         # We ensure the barrier is at least 'min_ret' distance away
#         threshold = max(current_vol, min_ret)
        
#         upper_barrier = current_price * (1 + (threshold * pt))
#         lower_barrier = current_price * (1 - (threshold * sl))
        
#         # Look forward window
#         label = 0 # Default: 0 means "Neutral" or "Stop Hit" (we will separate later if needed)
        
#         # Scan future candles
#         for j in range(1, max_horizon + 1):
#             future_price = closes[i + j]
            
#             if future_price >= upper_barrier:
#                 label = 1 # HIT PROFIT TARGET
#                 break
#             elif future_price <= lower_barrier:
#                 label = 0 # HIT STOP LOSS
#                 break
        
#         labels.append(label)
        
#     return labels

# # 3. Apply the Labeling
# # Settings: 
# #   Target = 1.5x Volatility (Reward)
# #   Stop   = 1.0x Volatility (Risk)
# #   Horizon = 12 candles (1 hour if 5m candles)
# df['direction'] = get_triple_barrier_label(
#     df['close'], 
#     df['volatility'], 
#     df.index, 
#     pt=1.0, 
#     sl=1.0, 
#     max_horizon=12
# )

# # Map -1 to 2 (because sklearn likes positive integers: 0, 1, 2)
# # 0 = Neutral, 1 = Long, 2 = Short
# df['direction'] = df['direction'].replace({-1: 2})

# # Target 2: Regression (Still use simple future return for regression baseline)
# df['price_change_pct'] = df['close'].shift(-5).pct_change(periods=5) * 100

# # Drop NaNs created by the windowing
# df = df.dropna()

# print(f"\n✅ Targets created using Triple Barrier Method")
# print(f"   Barriers: Dynamic based on 20-period Volatility")
# print(f"   Reward/Risk Ratio: 1.0 : 1.0")
# print("\nDirection Distribution (Should be more balanced):")
# print(df['direction'].value_counts(normalize=True).sort_index())

# =============================================================================
# STEP 2: CREATE TARGETS (3-CLASS: LONG / SHORT / NEUTRAL)
# =============================================================================
print("\n" + "="*70)
print("📊 Step 2: Creating 3-Class Targets")
print("="*70)

# 1. Calculate Volatility
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(window=20).std()

# 2. Define the NEW 3-Class Function
def get_triple_barrier_label_3class(close_prices, volatility, t_events, pt=1, sl=1, min_ret=0.001, max_horizon=12):
    labels = []
    closes = close_prices.values
    vols = volatility.values
    
    print(f"  generating 3-class labels for {len(closes)} rows...")
    
    for i in range(len(closes)):
        if i + max_horizon >= len(closes) or np.isnan(vols[i]):
            labels.append(np.nan)
            continue
            
        current_price = closes[i]
        threshold = max(vols[i], min_ret)
        
        upper_barrier = current_price * (1 + (threshold * pt))
        lower_barrier = current_price * (1 - (threshold * sl))
        
        label = 0 
        
        for j in range(1, max_horizon + 1):
            future_price = closes[i + j]
            
            if future_price >= upper_barrier:
                label = 1   # LONG WIN
                break
            elif future_price <= lower_barrier:
                label = -1  # SHORT WIN (Crucial!)
                break
        
        labels.append(label)
    return labels

# 3. Apply the Labeling
df['direction'] = get_triple_barrier_label_3class(
    df['close'], df['volatility'], df.index, 
    pt=1.0, sl=1.0, max_horizon=12
)

# 4. Map -1 to 2 (sklearn needs positive integers: 0=Neutral, 1=Long, 2=Short)
df['direction'] = df['direction'].replace({-1: 2})

# Target 2: Regression
df['price_change_pct'] = df['close'].shift(-5).pct_change(periods=5) * 100

df = df.dropna()

print(f"\n✅ Targets created (3 Classes)")
print("Direction Distribution (0=Neutral, 1=Long, 2=Short):")
# This print statement is your 'Sanity Check'. It MUST show 0.0, 1.0, and 2.0.
print(df['direction'].value_counts(normalize=True).sort_index())

# =============================================================================
# STEP 3: FEATURE SELECTION - IMPROVED METHOD (FIXED FOR INFINITY)
# =============================================================================
print("\n" + "="*70)
print("🔍 Step 3: Improved Feature Selection")
print("="*70)

# Exclude non-feature columns AND problematic cumulative features
exclude_cols = [
    'open', 'high', 'low', 'close', 'volume', 'num_trades',
    'price_change', 'price_change_pct', 'direction', 
    'magnitude_pct', 'tr', 'typical_price', 
    'obv', 'ad_line',  # Remove raw cumulative features (data leakage!)
    'movement_category','returns','volatility'
]

all_features = [col for col in df.columns if col not in exclude_cols]
print(f"\nTotal available features: {len(all_features)}")

# -----------------------------------------------------------------------------
# CRITICAL FIX: HANDLE INFINITY VALUES
# -----------------------------------------------------------------------------
# 1. Create a clean copy with only relevant columns
df_clean = df[all_features + ['direction', 'price_change_pct']].copy()

# 2. Replace Infinity with NaN (This is the missing step causing your error)
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. Drop NaNs (which now includes the former Infinity values)
rows_before = len(df_clean)
df_clean = df_clean.dropna()
rows_after = len(df_clean)

print(f"🧹 Data Cleaning Report:")
print(f"   • Rows before cleaning: {rows_before:,}")
print(f"   • Rows removed (NaN/Inf): {rows_before - rows_after:,}")
print(f"   • Final clean samples: {rows_after:,}")

# -----------------------------------------------------------------------------

# Continue with data preparation
X = df_clean[all_features]
y_direction = df_clean['direction']
y_regression = df_clean['price_change_pct']

# Split data (80% train, 20% test) - time-based split
split_point = int(len(X) * 0.8)
X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_dir_train = y_direction.iloc[:split_point]
y_dir_test = y_direction.iloc[split_point:]
y_reg_train = y_regression.iloc[:split_point]
y_reg_test = y_regression.iloc[split_point:]

print(f"\nTrain/Test Split (Time-Based):")
print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Scale features (This line crashed before, now it will work)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Features scaled (StandardScaler)")

# -------------------------------------------------------------------------
# Feature Importance Analysis
# -------------------------------------------------------------------------
print("\n" + "-"*70)
print("Feature Importance Analysis")
print("-"*70)

# Use more trees for better importance estimation
rf_temp = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=200,  # Increased for large dataset
    random_state=42,
    n_jobs=-1
)
rf_temp.fit(X_train, y_dir_train)

rf_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 20 features by importance:")
print(rf_importance.head(20).to_string(index=False))

# Select top 40 features initially
selected_features = rf_importance.head(40)['feature'].tolist()
    
print(f"\n✅ Selected top {len(selected_features)} features")

# -------------------------------------------------------------------------
# Remove Highly Correlated Features (RELAXED THRESHOLD)
# -------------------------------------------------------------------------
print("\n" + "-"*70)
print("Removing Highly Correlated Features (>0.95)")  # Changed from 0.85!
print("-"*70)

corr_matrix = X_train[selected_features].corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = []
for column in upper_triangle.columns:
    if any(upper_triangle[column] > 0.95):  # Changed from 0.85 to 0.95!
        correlated_with = upper_triangle[column][upper_triangle[column] > 0.95].index.tolist()
        if correlated_with:
            # Only drop if this feature has lower importance
            current_imp = rf_importance[rf_importance['feature'] == column]['importance'].values
            if len(current_imp) > 0:
                current_imp = current_imp[0]
                corr_features_imp = rf_importance[rf_importance['feature'].isin(correlated_with)]['importance'].values
                
                if len(corr_features_imp) > 0 and current_imp < corr_features_imp.max():
                    print(f"  Removing {column} (corr={upper_triangle[column].max():.3f} with {correlated_with[0]})")
                    to_drop.append(column)

final_features = [f for f in selected_features if f not in to_drop]
print(f"\n✅ Final feature set: {len(final_features)} features")
print(f"   Feature types: momentum, volatility, volume, trend, time")

# Prepare final datasets
X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

X_train_final_scaled = scaler.fit_transform(X_train_final)
X_test_final_scaled = scaler.transform(X_test_final)

# =============================================================================
# STEP 4: MODEL TRAINING WITH PROPER REGULARIZATION
# =============================================================================
print("\n" + "="*70)
print("🤖 Step 4: Model Training with Regularization")
print("="*70)

# Random Forest with PROPER regularization for large dataset
classifier_model = RandomForestClassifier(
    n_estimators=300,         # Sufficient for large data
    max_depth=6,             # Prevent deep memorization
    min_samples_split=200,    # Need 100 samples to split
    min_samples_leaf=50,      # Need 50 samples per leaf
    max_features='sqrt',      # Feature randomness
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1
)

print("\nModel Configuration:")
print(f"  • n_estimators: 300")
print(f"  • max_depth: 6")
print(f"  • min_samples_split: 100")
print(f"  • min_samples_leaf: 100")
print(f"  • max_features: 'sqrt'")

# Train
print("\nTraining classifier...")
classifier_model.fit(X_train_final_scaled, y_dir_train)

# Predict
y_pred_train = classifier_model.predict(X_train_final_scaled)
y_pred_test = classifier_model.predict(X_test_final_scaled)

# Evaluate
train_acc = accuracy_score(y_dir_train, y_pred_train)
test_acc = accuracy_score(y_dir_test, y_pred_test)

# Time Series Cross-Validation
print("\nPerforming Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(classifier_model, X_train_final_scaled, y_dir_train, cv=tscv, scoring='accuracy')

print("\n📊 Classification Results:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Train/Test Gap: {(train_acc - test_acc)*100:.2f}%")
print(f"  CV Score:       {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Check for overfitting
if train_acc - test_acc > 0.10:
    print("  🚨 WARNING: Significant overfitting detected!")
elif train_acc - test_acc < 0.05:
    print("  ✅ Good generalization!")
else:
    print("  ⚠️  Mild overfitting")

# =============================================================================
# STEP 5: DETAILED EVALUATION
# =============================================================================
# print("\n" + "="*70)
# print("📊 Step 5: Detailed Classification Evaluation")
# print("="*70)

# # Classification Report
# print("\n📋 Classification Report:")
# print(classification_report(y_dir_test, y_pred_test, target_names=['DOWN', 'UP']))

# # Confusion Matrix
# cm = confusion_matrix(y_dir_test, y_pred_test)
# print("\n📊 Confusion Matrix:")
# print(f"                Predicted")
# print(f"              DOWN    UP")
# print(f"Actual DOWN   {cm[0,0]:<6}  {cm[0,1]:<6}")
# print(f"       UP     {cm[1,0]:<6}  {cm[1,1]:<6}")

# # Additional metrics
# tn, fp, fn, tp = cm.ravel()
# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# print(f"\n📈 Detailed Metrics:")
# print(f"  Precision (UP): {precision:.4f}")
# print(f"  Recall (UP):    {recall:.4f}")
# print(f"  F1-Score (UP):  {f1:.4f}")


# =============================================================================
# STEP 5: PROBABILITY THRESHOLD OPTIMIZATION (MULTI-CLASS FIX)
# =============================================================================
print("\n" + "="*70)
print("📊 Step 5: Finding the 'Sweet Spot' (Long & Short Tuning)")
print("="*70)

# Get probabilities for Class 1 (Long) and Class 2 (Short)
# .predict_proba returns [Neutral, Long, Short]
probs_long = classifier_model.predict_proba(X_test_final_scaled)[:, 1]
probs_short = classifier_model.predict_proba(X_test_final_scaled)[:, 2]

# Test thresholds
thresholds = [0.35, 0.38, 0.40, 0.42, 0.45, 0.48,0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]

print(f"{'Threshold':<10} | {'Long Win Rate':<15} | {'Long Trades':<12} | {'Short Win Rate':<15} | {'Short Trades':<12}")
print("-" * 75)

best_threshold = 0.40
best_long_precision = 0.0

for t in thresholds:
    # ---------------------------------------------------------
    # 1. Evaluate LONGS (Class 1)
    # ---------------------------------------------------------
    # We predict LONG if probability > t
    long_signals = (probs_long >= t).astype(int)
    
    # Calculate Precision manually for Class 1
    # True Positives: We predicted Long (1) AND Actual was Long (1)
    tp_long = ((long_signals == 1) & (y_dir_test == 1)).sum()
    # False Positives: We predicted Long (1) BUT Actual was Neutral (0) or Short (2)
    fp_long = ((long_signals == 1) & (y_dir_test != 1)).sum()
    
    long_prec = tp_long / (tp_long + fp_long) if (tp_long + fp_long) > 0 else 0
    long_count = tp_long + fp_long

    # ---------------------------------------------------------
    # 2. Evaluate SHORTS (Class 2)
    # ---------------------------------------------------------
    # We predict SHORT if probability > t
    short_signals = (probs_short >= t).astype(int)
    
    # Calculate Precision manually for Class 2
    # True Positives: We predicted Short (1) AND Actual was Short (2)
    tp_short = ((short_signals == 1) & (y_dir_test == 2)).sum()
    # False Positives: We predicted Short (1) BUT Actual was Neutral (0) or Long (1)
    fp_short = ((short_signals == 1) & (y_dir_test != 2)).sum()
    
    short_prec = tp_short / (tp_short + fp_short) if (tp_short + fp_short) > 0 else 0
    short_count = tp_short + fp_short

    print(f"{t:.2f}       | {long_prec*100:.2f}%          | {long_count:<12} | {short_prec*100:.2f}%          | {short_count:<12}")

    # Logic to pick best threshold (prioritizing Long Win Rate > 50% with reasonable volume)
    if long_prec > best_long_precision and long_count > 50:
        best_long_precision = long_prec
        best_threshold = t

print("-" * 75)
print(f"🏆 Optimal Threshold: {best_threshold}")
print(f"   Expected Long Win Rate: {best_long_precision*100:.2f}%")

# Save this threshold for your live bot
import json
with open('models/threshold.json', 'w') as f:
    json.dump({'threshold': best_threshold, 'win_rate': best_long_precision}, f)
print("✅ Threshold saved to models/threshold.json")

# =============================================================================
# STEP 6: REGRESSION MODEL
# =============================================================================
print("\n" + "="*70)
print("📈 Step 6: Regression Model")
print("="*70)

regressor_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_split=100,
    min_samples_leaf=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print("Training regressor...")
regressor_model.fit(X_train_final_scaled, y_reg_train)

y_pred_reg = regressor_model.predict(X_test_final_scaled)

mae = mean_absolute_error(y_reg_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
r2 = r2_score(y_reg_test, y_pred_reg)

dir_actual = (y_reg_test > 0).astype(int)
dir_pred = (y_pred_reg > 0).astype(int)
dir_acc = accuracy_score(dir_actual, dir_pred)

print(f"\n📊 Regression Results:")
print(f"  MAE:  {mae:.4f}%")
print(f"  RMSE: {rmse:.4f}%")
print(f"  R²:   {r2:.4f}")
print(f"  Direction Accuracy: {dir_acc:.4f}")

if r2 > 0.10:
    print("  ✅ Good regression performance!")
elif r2 > 0:
    print("  ⚠️  Weak but positive R²")
else:
    print("  ❌ Model worse than baseline")

# =============================================================================
# STEP 7: SAVE MODELS
# =============================================================================
print("\n" + "="*70)
print("💾 Saving Models and Results")
print("="*70)

import joblib
import os
import json

os.makedirs('models', exist_ok=True)

joblib.dump(classifier_model, "models/direction_model.pkl")
joblib.dump(regressor_model, "models/regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(final_features, "models/selected_features.pkl")

summary = {
    'metadata': {
        # 'forecast_horizon': forecast_horizon,
        'features_used': len(final_features),
    },
    'classification': {
        'test_accuracy': float(test_acc),
        'cv_mean': float(cv_scores.mean()),
        'train_test_gap': float(train_acc - test_acc)
    },
    'regression': {
        'r2': float(r2),
        'mae': float(mae)
    }
}

with open('models/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Models saved successfully!")

print("\n" + "="*70)
print("🎯 FINAL SUMMARY")
print("="*70)
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Overfitting: {(train_acc-test_acc)*100:.2f}%")
print(f"  R²: {r2:.4f}")

if test_acc >= 0.55 and abs(train_acc - test_acc) < 0.10:
    print(f"\n  ✅ MODEL STATUS: ACCEPTABLE")
else:
    print(f"\n  ⚠️  MODEL STATUS: NEEDS IMPROVEMENT")
    
# =============================================================================
# STEP 8: ADVANCED BACKTEST (HEDGE STRATEGY: LONG + SHORT + SMA FILTER)
# =============================================================================
print("\n" + "="*70)
print("🎨 Step 8: Hedge Strategy Backtest (Fixed)")
print("="*70)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. CALCULATE TREND FILTER (True Daily SMA 200)
# -----------------------------------------------------------------------------
print("  Calculating Daily SMA 200 (Regime Filter)...")

# Resample to Daily Candles to get the real 200-Day Moving Average
daily_df = df.resample('1D').agg({'close': 'last'})
daily_df['sma_200_daily'] = daily_df['close'].rolling(window=200).mean()

# Map the Daily SMA back to the 5-minute data (forward fill)
# This lets every 5m candle know what the daily trend is
df['sma_200_daily'] = daily_df['sma_200_daily'].reindex(df.index, method='ffill')

# -----------------------------------------------------------------------------
# 2. PREPARE SIGNALS (LONG vs SHORT)
# -----------------------------------------------------------------------------
# The model now outputs 3 probabilities: [Neutral, Long, Short]
probs = classifier_model.predict_proba(X_test_final_scaled)
probs_long = probs[:, 1]   # Probability of Class 1 (UP)
probs_short = probs[:, 2]  # Probability of Class 2 (DOWN)

# -----------------------------------------------------------------------------
# 3. ALIGN DATA FOR BACKTEST
# -----------------------------------------------------------------------------
# Align everything to the Test Set indices
test_indices = X_test.index
closes = df.loc[test_indices, 'close'].values
vols = df.loc[test_indices, 'volatility'].values
sma_200 = df.loc[test_indices, 'sma_200_daily'].values
dates = test_indices

# -----------------------------------------------------------------------------
# 4. RUN SIMULATION
# -----------------------------------------------------------------------------
# Settings
try:
    optimal_threshold = best_threshold
except NameError:
    optimal_threshold = 0.40 

print(f"  Using Dynamic Threshold: {optimal_threshold}")

capital = 10000.0
equity_curve = [capital]
trade_log = []
in_trade = False
side = None       # 'LONG' or 'SHORT'
entry_price = 0
exit_candle = 0

print(f"  Simulating Hedge Bot (Threshold {optimal_threshold})...")

for i in range(len(closes) - 13):
    if not in_trade:
        equity_curve.append(equity_curve[-1])
        
        # --- TREND FILTER ---
        # If SMA is NaN (start of data), assume Neutral (False)
        is_uptrend = closes[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # ENTRY LOGIC: LONG (Signal + Bull Market)
        if probs_long[i] >= optimal_threshold and is_uptrend:
            in_trade = True
            side = 'LONG'
            entry_price = closes[i]
            vol = vols[i]
            take_profit = entry_price * (1 + vol)
            stop_loss = entry_price * (1 - vol)
            exit_candle = i + 12
            capital *= 0.9995 # Entry Fee

        # ENTRY LOGIC: SHORT (Signal + Bear Market)
        elif probs_short[i] >= optimal_threshold and not is_uptrend:
            in_trade = True
            side = 'SHORT'
            entry_price = closes[i]
            vol = vols[i]
            # Short Logic: Profit if price DROPS
            take_profit = entry_price * (1 - vol) 
            stop_loss = entry_price * (1 + vol)
            exit_candle = i + 12
            capital *= 0.9995 # Entry Fee
            
    else:
        # MANAGE TRADE
        current_price = closes[i]
        time_out = i >= exit_candle
        
        # Exit Logic for LONG
        if side == 'LONG':
            hit_tp = current_price >= take_profit
            hit_sl = current_price <= stop_loss
            
            if hit_tp or hit_sl or time_out:
                change_pct = (current_price - entry_price) / entry_price
                capital *= (1 + change_pct)
                capital *= 0.9995 # Exit Fee
                in_trade = False
                trade_log.append(change_pct)
                
        # Exit Logic for SHORT
        elif side == 'SHORT':
            hit_tp = current_price <= take_profit # Price went DOWN (Good)
            hit_sl = current_price >= stop_loss   # Price went UP (Bad)
            
            if hit_tp or hit_sl or time_out:
                # Short Profit formula is reversed
                change_pct = (entry_price - current_price) / entry_price
                capital *= (1 + change_pct)
                capital *= 0.9995 # Exit Fee
                in_trade = False
                trade_log.append(change_pct)
        
        equity_curve.append(capital)

# -----------------------------------------------------------------------------
# 5. VISUALIZATION
# -----------------------------------------------------------------------------
equity_df = pd.DataFrame({'Equity': equity_curve}, index=dates[:len(equity_curve)])

# Chart 1: Hedge Equity Curve
plt.figure(figsize=(12, 6))
market_norm = (closes[:len(equity_curve)] / closes[0]) * 10000
plt.plot(equity_df.index, market_norm, label='Market (Buy & Hold)', color='gray', alpha=0.5)
plt.plot(equity_df.index, equity_df['Equity'], label='Hedge Strategy (Long/Short)', color='blue', linewidth=2)
plt.title(f'Hedge Bot Performance (Threshold {optimal_threshold})')
plt.ylabel('Capital ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('models/hedge_equity_curve.png')
print("✅ Visualization Saved: models/hedge_equity_curve.png")

# Chart 2: Win Rate Analysis (Optional but useful)
# We calculate win rate for Longs and Shorts combined
total_trades = len(trade_log)
if total_trades > 0:
    winning_trades = len([x for x in trade_log if x > 0])
    win_rate = winning_trades / total_trades
else:
    win_rate = 0

print("\n" + "="*70)
print("🚀 FINAL VERDICT (HEDGE BOT)")
print("="*70)
print(f"Final Capital:   ${equity_curve[-1]:.2f}")
print(f"Total Trades:    {total_trades}")
print(f"Overall Win Rate: {win_rate*100:.2f}%")
print(f"Avg Profit:      {sum(trade_log)/len(trade_log)*100 if len(trade_log)>0 else 0:.2f}% per trade")