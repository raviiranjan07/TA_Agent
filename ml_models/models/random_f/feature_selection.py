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
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD CONFIGURATION FROM .env
# =============================================================================
load_dotenv()

# Model selection: "both", "rf", or "xgb"
MODEL_TYPE = os.getenv('MODEL_TYPE', 'both').lower()

# Random Forest hyperparameters
RF_N_ESTIMATORS = int(os.getenv('RF_N_ESTIMATORS', 300))
RF_MAX_DEPTH = int(os.getenv('RF_MAX_DEPTH', 6))
RF_MIN_SAMPLES_SPLIT = int(os.getenv('RF_MIN_SAMPLES_SPLIT', 200))
RF_MIN_SAMPLES_LEAF = int(os.getenv('RF_MIN_SAMPLES_LEAF', 50))

# XGBoost hyperparameters
XGB_N_ESTIMATORS = int(os.getenv('XGB_N_ESTIMATORS', 300))
XGB_MAX_DEPTH = int(os.getenv('XGB_MAX_DEPTH', 4))
XGB_LEARNING_RATE = float(os.getenv('XGB_LEARNING_RATE', 0.05))
XGB_SUBSAMPLE = float(os.getenv('XGB_SUBSAMPLE', 0.8))
XGB_COLSAMPLE_BYTREE = float(os.getenv('XGB_COLSAMPLE_BYTREE', 0.8))

# Backtest configuration
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10000))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.02))
TRADING_FEE = float(os.getenv('TRADING_FEE', 0.0005))

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

# DEBUG: Check what we loaded
print(f"\n  DEBUG: Total rows loaded: {len(df)}")
print(f"  DEBUG: Columns available: {len(df.columns)}")
print(f"  DEBUG: 'volatility_20' in columns: {'volatility_20' in df.columns}")
print(f"  DEBUG: 'close' column NaN count: {df['close'].isna().sum()}")

# 1. Use existing volatility from feature engineering (or calculate if missing)
if 'volatility_20' in df.columns:
    print(f"  DEBUG: volatility_20 NaN count: {df['volatility_20'].isna().sum()}")
    print(f"  DEBUG: volatility_20 sample values: {df['volatility_20'].head(3).tolist()}")
    df['volatility'] = df['volatility_20']
    print("  Using existing volatility_20 column")
else:
    print("  WARNING: volatility_20 not found, calculating from returns...")
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    print("  Calculated new volatility column")

print(f"  DEBUG: volatility NaN count after assignment: {df['volatility'].isna().sum()}")

# Drop rows where volatility is NaN (warm-up period)
rows_before = len(df)
df = df.dropna(subset=['volatility', 'close'])
print(f"  Dropped {rows_before - len(df)} rows with NaN volatility")
print(f"  Remaining rows: {len(df):,}")

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

# Target 2: Regression (FIXED: correct future return calculation)
# Calculate: "What % will price change in 5 candles from now?"
df['price_change_pct'] = (df['close'].shift(-5) - df['close']) / df['close'] * 100

# Only drop rows where TARGET columns are NaN (not all columns!)
rows_before_target_drop = len(df)
df = df.dropna(subset=['direction', 'price_change_pct'])
print(f"  Dropped {rows_before_target_drop - len(df)} rows with NaN in targets")
print(f"  Remaining rows after target creation: {len(df):,}")

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
    # Raw OHLCV data
    'open', 'high', 'low', 'close', 'volume', 'num_trades',
    # Target variables
    'price_change', 'price_change_pct', 'direction',
    # Intermediate/helper columns
    'magnitude_pct', 'tr', 'typical_price',
    'movement_category', 'returns', 'volatility',
    # Cumulative features (potential data leakage)
    'vwap',  # Uses cumsum across entire history
    # Intermediate streak feature (keep consecutive_up/down instead)
    'consecutive_streak',
]

all_features = [col for col in df.columns if col not in exclude_cols]
print(f"\nTotal available features: {len(all_features)}")

# -----------------------------------------------------------------------------
# CRITICAL FIX: HANDLE INFINITY VALUES
# -----------------------------------------------------------------------------
# 1. Create a clean copy with only relevant columns
df_clean = df[all_features + ['direction', 'price_change_pct']].copy()

# 2. Replace Infinity with NaN
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# DEBUG: Find which columns have most NaN values
nan_counts = df_clean.isna().sum()
problematic_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
print(f"\n  DEBUG: Columns with NaN values ({len(problematic_cols)} total):")
print(problematic_cols.head(20))

# 3. Instead of dropping all NaN rows, fill NaN with column median for features
# (Only drop rows where TARGET is NaN, which we already did)
rows_before = len(df_clean)

# Fill NaN in feature columns with median (safer than dropping all rows)
for col in all_features:
    if df_clean[col].isna().any():
        median_val = df_clean[col].median()
        if pd.isna(median_val):
            median_val = 0  # If all values are NaN, use 0
        df_clean[col].fillna(median_val, inplace=True)

# Now check if any NaN remain (should only be in targets, which are already clean)
remaining_nan = df_clean.isna().sum().sum()
print(f"  DEBUG: Remaining NaN after filling: {remaining_nan}")

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
print("🤖 Step 4: Model Training")
print("="*70)
print(f"  Configuration: MODEL_TYPE = '{MODEL_TYPE}'")

# Initialize variables
rf_classifier = None
xgb_classifier = None
rf_train_acc = rf_test_acc = 0
xgb_train_acc = xgb_test_acc = 0
rf_pred_train = rf_pred_test = None
xgb_pred_train = xgb_pred_test = None

# -----------------------------------------------------------------------------
# 4A: RANDOM FOREST CLASSIFIER (if enabled)
# -----------------------------------------------------------------------------
if MODEL_TYPE in ['both', 'rf']:
    print("\n" + "-"*70)
    print("Training Random Forest Classifier...")
    print(f"  Params: n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}")
    print("-"*70)

    rf_classifier = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    rf_classifier.fit(X_train_final_scaled, y_dir_train)

    rf_pred_train = rf_classifier.predict(X_train_final_scaled)
    rf_pred_test = rf_classifier.predict(X_test_final_scaled)

    rf_train_acc = accuracy_score(y_dir_train, rf_pred_train)
    rf_test_acc = accuracy_score(y_dir_test, rf_pred_test)

    print(f"  RF Train Accuracy: {rf_train_acc:.4f}")
    print(f"  RF Test Accuracy:  {rf_test_acc:.4f}")
    print(f"  RF Train/Test Gap: {(rf_train_acc - rf_test_acc)*100:.2f}%")
else:
    print("\n⏭️  Skipping Random Forest (MODEL_TYPE = 'xgb')")

# -----------------------------------------------------------------------------
# 4B: XGBOOST CLASSIFIER (if enabled)
# -----------------------------------------------------------------------------
if MODEL_TYPE in ['both', 'xgb']:
    print("\n" + "-"*70)
    print("Training XGBoost Classifier...")
    print(f"  Params: n_estimators={XGB_N_ESTIMATORS}, max_depth={XGB_MAX_DEPTH}, lr={XGB_LEARNING_RATE}")
    print("-"*70)

    xgb_classifier = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    xgb_classifier.fit(X_train_final_scaled, y_dir_train)

    xgb_pred_train = xgb_classifier.predict(X_train_final_scaled)
    xgb_pred_test = xgb_classifier.predict(X_test_final_scaled)

    xgb_train_acc = accuracy_score(y_dir_train, xgb_pred_train)
    xgb_test_acc = accuracy_score(y_dir_test, xgb_pred_test)

    print(f"  XGB Train Accuracy: {xgb_train_acc:.4f}")
    print(f"  XGB Test Accuracy:  {xgb_test_acc:.4f}")
    print(f"  XGB Train/Test Gap: {(xgb_train_acc - xgb_test_acc)*100:.2f}%")
else:
    print("\n⏭️  Skipping XGBoost (MODEL_TYPE = 'rf')")

# -----------------------------------------------------------------------------
# 4C: SELECT BEST MODEL
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("Model Selection")
print("-"*70)

if MODEL_TYPE == 'both':
    print(f"\n{'Model':<15} | {'Train Acc':<12} | {'Test Acc':<12} | {'Overfit':<10}")
    print("-" * 55)
    print(f"{'Random Forest':<15} | {rf_train_acc:.4f}       | {rf_test_acc:.4f}       | {(rf_train_acc-rf_test_acc)*100:.2f}%")
    print(f"{'XGBoost':<15} | {xgb_train_acc:.4f}       | {xgb_test_acc:.4f}       | {(xgb_train_acc-xgb_test_acc)*100:.2f}%")

    # Select best model based on test accuracy with minimal overfitting
    rf_score = rf_test_acc - abs(rf_train_acc - rf_test_acc) * 0.5  # Penalize overfitting
    xgb_score = xgb_test_acc - abs(xgb_train_acc - xgb_test_acc) * 0.5

    if xgb_score > rf_score:
        classifier_model = xgb_classifier
        y_pred_train = xgb_pred_train
        y_pred_test = xgb_pred_test
        train_acc = xgb_train_acc
        test_acc = xgb_test_acc
        best_model_name = "XGBoost"
        print(f"\n🏆 Best Model: XGBoost (score: {xgb_score:.4f})")
    else:
        classifier_model = rf_classifier
        y_pred_train = rf_pred_train
        y_pred_test = rf_pred_test
        train_acc = rf_train_acc
        test_acc = rf_test_acc
        best_model_name = "Random Forest"
        print(f"\n🏆 Best Model: Random Forest (score: {rf_score:.4f})")

elif MODEL_TYPE == 'rf':
    classifier_model = rf_classifier
    y_pred_train = rf_pred_train
    y_pred_test = rf_pred_test
    train_acc = rf_train_acc
    test_acc = rf_test_acc
    best_model_name = "Random Forest"
    print(f"\n🏆 Selected Model: Random Forest (single model mode)")

elif MODEL_TYPE == 'xgb':
    classifier_model = xgb_classifier
    y_pred_train = xgb_pred_train
    y_pred_test = xgb_pred_test
    train_acc = xgb_train_acc
    test_acc = xgb_test_acc
    best_model_name = "XGBoost"
    print(f"\n🏆 Selected Model: XGBoost (single model mode)")

else:
    raise ValueError(f"Invalid MODEL_TYPE: '{MODEL_TYPE}'. Must be 'both', 'rf', or 'xgb'")

# Time Series Cross-Validation on best model
print("\nPerforming Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(classifier_model, X_train_final_scaled, y_dir_train, cv=tscv, scoring='accuracy')

print(f"\n📊 Best Model ({best_model_name}) Results:")
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
# STEP 5: DETAILED 3-CLASS EVALUATION
# =============================================================================
print("\n" + "="*70)
print("📊 Step 5: Detailed 3-Class Evaluation")
print("="*70)

# Classification Report (3-Class)
print("\n📋 Classification Report (3-Class):")
print(classification_report(y_dir_test, y_pred_test,
      target_names=['Neutral', 'Long', 'Short'], zero_division=0))

# Confusion Matrix (3-Class)
cm = confusion_matrix(y_dir_test, y_pred_test)
print("📊 Confusion Matrix:")
print(f"                    Predicted")
print(f"               Neutral  Long  Short")
for i, label in enumerate(['Neutral', 'Long   ', 'Short  ']):
    row = "  ".join(f"{cm[i,j]:<6}" for j in range(min(3, cm.shape[1])))
    print(f"Actual {label}  {row}")

# Per-class metrics
print(f"\n📈 Per-Class Accuracy:")
for i, label in enumerate(['Neutral', 'Long', 'Short']):
    if i < cm.shape[0]:
        class_total = cm[i, :].sum()
        class_correct = cm[i, i] if i < cm.shape[1] else 0
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {label}: {class_acc*100:.1f}% ({class_correct}/{class_total})")


# =============================================================================
# STEP 5B: PROBABILITY THRESHOLD OPTIMIZATION (BALANCED FOR LONG & SHORT)
# =============================================================================
print("\n" + "="*70)
print("📊 Step 5b: Threshold Optimization (Long & Short)")
print("="*70)

# Get probabilities for Class 1 (Long) and Class 2 (Short)
probs_long = classifier_model.predict_proba(X_test_final_scaled)[:, 1]
probs_short = classifier_model.predict_proba(X_test_final_scaled)[:, 2]

# Test thresholds
thresholds = [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]

print(f"{'Threshold':<10} | {'Long Win%':<10} | {'Long #':<8} | {'Short Win%':<10} | {'Short #':<8} | {'Combined':<10}")
print("-" * 75)

best_threshold_long = 0.40
best_threshold_short = 0.40
best_long_precision = 0.0
best_short_precision = 0.0
best_combined_score = 0.0

threshold_results = []

for t in thresholds:
    # Evaluate LONGS
    long_signals = (probs_long >= t).astype(int)
    tp_long = ((long_signals == 1) & (y_dir_test == 1)).sum()
    fp_long = ((long_signals == 1) & (y_dir_test != 1)).sum()
    long_prec = tp_long / (tp_long + fp_long) if (tp_long + fp_long) > 0 else 0
    long_count = tp_long + fp_long

    # Evaluate SHORTS
    short_signals = (probs_short >= t).astype(int)
    tp_short = ((short_signals == 1) & (y_dir_test == 2)).sum()
    fp_short = ((short_signals == 1) & (y_dir_test != 2)).sum()
    short_prec = tp_short / (tp_short + fp_short) if (tp_short + fp_short) > 0 else 0
    short_count = tp_short + fp_short

    # Combined score: weighted average of precision * sqrt(trade_count) for balance
    long_score = long_prec * np.sqrt(long_count) if long_count > 20 else 0
    short_score = short_prec * np.sqrt(short_count) if short_count > 20 else 0
    combined = (long_score + short_score) / 2

    print(f"{t:.2f}       | {long_prec*100:>6.1f}%   | {long_count:<8} | {short_prec*100:>6.1f}%   | {short_count:<8} | {combined:.2f}")

    threshold_results.append({
        'threshold': t,
        'long_prec': long_prec, 'long_count': long_count,
        'short_prec': short_prec, 'short_count': short_count,
        'combined': combined
    })

    # Track best for each
    if long_prec > best_long_precision and long_count > 50:
        best_long_precision = long_prec
        best_threshold_long = t
    if short_prec > best_short_precision and short_count > 50:
        best_short_precision = short_prec
        best_threshold_short = t
    if combined > best_combined_score:
        best_combined_score = combined
        best_threshold = t

print("-" * 75)
print(f"🏆 Best Long Threshold:  {best_threshold_long} (Win Rate: {best_long_precision*100:.1f}%)")
print(f"🏆 Best Short Threshold: {best_threshold_short} (Win Rate: {best_short_precision*100:.1f}%)")
print(f"🏆 Best Combined:        {best_threshold}")

# Save thresholds
import json
threshold_config = {
    'threshold_long': best_threshold_long,
    'threshold_short': best_threshold_short,
    'threshold_combined': best_threshold,
    'long_win_rate': best_long_precision,
    'short_win_rate': best_short_precision
}
with open('models/threshold.json', 'w') as f:
    json.dump(threshold_config, f, indent=2)
print("✅ Thresholds saved to models/threshold.json")

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
# STEP 8: ADVANCED BACKTEST WITH RISK METRICS & POSITION SIZING
# =============================================================================
print("\n" + "="*70)
print("🎨 Step 8: Advanced Backtest (Risk Metrics + Position Sizing)")
print("="*70)

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. CALCULATE TREND FILTER (True Daily SMA 200)
# -----------------------------------------------------------------------------
print("  Calculating Daily SMA 200 (Regime Filter)...")

daily_df = df.resample('1D').agg({'close': 'last'})
daily_df['sma_200_daily'] = daily_df['close'].rolling(window=200).mean()
df['sma_200_daily'] = daily_df['sma_200_daily'].reindex(df.index, method='ffill')

# -----------------------------------------------------------------------------
# 2. PREPARE SIGNALS
# -----------------------------------------------------------------------------
probs = classifier_model.predict_proba(X_test_final_scaled)
probs_long = probs[:, 1]
probs_short = probs[:, 2]

test_indices = X_test.index
closes = df.loc[test_indices, 'close'].values
vols = df.loc[test_indices, 'volatility'].values
sma_200 = df.loc[test_indices, 'sma_200_daily'].values
dates = test_indices

# -----------------------------------------------------------------------------
# 3. BACKTEST WITH POSITION SIZING
# -----------------------------------------------------------------------------
# Use separate thresholds for Long and Short
threshold_long = best_threshold_long
threshold_short = best_threshold_short

print(f"  Long Threshold:  {threshold_long}")
print(f"  Short Threshold: {threshold_short}")

# Position sizing parameters (loaded from .env)
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"  Risk Per Trade:  {RISK_PER_TRADE*100:.1f}%")
print(f"  Trading Fee:     {TRADING_FEE*100:.3f}%")

capital = INITIAL_CAPITAL
equity_curve = [capital]
trade_log = []
long_trades = []
short_trades = []

in_trade = False
side = None
entry_price = 0
position_size = 0
exit_candle = 0

print(f"  Simulating with 2% Risk Per Trade...")

for i in range(len(closes) - 13):
    if not in_trade:
        equity_curve.append(capital)

        is_uptrend = closes[i] > sma_200[i] if not np.isnan(sma_200[i]) else False

        # LONG ENTRY
        if probs_long[i] >= threshold_long and is_uptrend:
            in_trade = True
            side = 'LONG'
            entry_price = closes[i]
            vol = max(vols[i], 0.001)  # Minimum volatility

            # Position sizing: Risk 2% of capital
            stop_distance = entry_price * vol
            position_size = (capital * RISK_PER_TRADE) / stop_distance
            position_value = position_size * entry_price

            take_profit = entry_price * (1 + vol * 1.5)  # 1.5:1 R:R
            stop_loss = entry_price * (1 - vol)
            exit_candle = i + 12

            # Entry fee
            capital -= position_value * TRADING_FEE

        # SHORT ENTRY
        elif probs_short[i] >= threshold_short and not is_uptrend:
            in_trade = True
            side = 'SHORT'
            entry_price = closes[i]
            vol = max(vols[i], 0.001)

            stop_distance = entry_price * vol
            position_size = (capital * RISK_PER_TRADE) / stop_distance
            position_value = position_size * entry_price

            take_profit = entry_price * (1 - vol * 1.5)
            stop_loss = entry_price * (1 + vol)
            exit_candle = i + 12

            capital -= position_value * TRADING_FEE

    else:
        current_price = closes[i]
        time_out = i >= exit_candle

        if side == 'LONG':
            hit_tp = current_price >= take_profit
            hit_sl = current_price <= stop_loss

            if hit_tp or hit_sl or time_out:
                pnl = (current_price - entry_price) * position_size
                capital += pnl
                capital -= abs(pnl) * TRADING_FEE  # Exit fee

                trade_return = (current_price - entry_price) / entry_price
                trade_log.append(trade_return)
                long_trades.append(trade_return)
                in_trade = False

        elif side == 'SHORT':
            hit_tp = current_price <= take_profit
            hit_sl = current_price >= stop_loss

            if hit_tp or hit_sl or time_out:
                pnl = (entry_price - current_price) * position_size
                capital += pnl
                capital -= abs(pnl) * TRADING_FEE

                trade_return = (entry_price - current_price) / entry_price
                trade_log.append(trade_return)
                short_trades.append(trade_return)
                in_trade = False

        equity_curve.append(capital)

# -----------------------------------------------------------------------------
# 4. CALCULATE RISK METRICS
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("Risk Metrics Calculation")
print("-"*70)

equity_array = np.array(equity_curve)
returns_array = np.diff(equity_array) / equity_array[:-1]

# Total Return
total_return = (equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# Sharpe Ratio (annualized, assuming 15-min candles = 35040 per year)
if len(returns_array) > 0 and np.std(returns_array) > 0:
    sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(35040)
else:
    sharpe_ratio = 0

# Max Drawdown
rolling_max = np.maximum.accumulate(equity_array)
drawdowns = (equity_array - rolling_max) / rolling_max
max_drawdown = np.min(drawdowns) * 100

# Calmar Ratio (annualized return / max drawdown)
annualized_return = total_return * (35040 / len(equity_curve)) if len(equity_curve) > 0 else 0
calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

# Win Rate
total_trades = len(trade_log)
winning_trades = len([x for x in trade_log if x > 0])
win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

# Profit Factor
gross_profit = sum([x for x in trade_log if x > 0])
gross_loss = abs(sum([x for x in trade_log if x < 0]))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

# Average Win / Average Loss
avg_win = np.mean([x for x in trade_log if x > 0]) * 100 if winning_trades > 0 else 0
avg_loss = np.mean([x for x in trade_log if x < 0]) * 100 if (total_trades - winning_trades) > 0 else 0

# Long vs Short breakdown
long_wins = len([x for x in long_trades if x > 0])
short_wins = len([x for x in short_trades if x > 0])
long_wr = long_wins / len(long_trades) * 100 if len(long_trades) > 0 else 0
short_wr = short_wins / len(short_trades) * 100 if len(short_trades) > 0 else 0

# -----------------------------------------------------------------------------
# 5. VISUALIZATION
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Equity Curve
ax1 = axes[0, 0]
market_norm = (closes[:len(equity_curve)] / closes[0]) * INITIAL_CAPITAL
ax1.plot(dates[:len(equity_curve)], market_norm, label='Buy & Hold', color='gray', alpha=0.5)
ax1.plot(dates[:len(equity_curve)], equity_curve, label='Strategy', color='blue', linewidth=2)
ax1.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Equity Curve')
ax1.set_ylabel('Capital ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2: Drawdown
ax2 = axes[0, 1]
ax2.fill_between(dates[:len(drawdowns)], drawdowns * 100, 0, color='red', alpha=0.3)
ax2.set_title('Drawdown (%)')
ax2.set_ylabel('Drawdown %')
ax2.grid(True, alpha=0.3)

# Chart 3: Trade Returns Distribution
ax3 = axes[1, 0]
if len(trade_log) > 0:
    ax3.hist([x * 100 for x in trade_log], bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--')
ax3.set_title('Trade Returns Distribution')
ax3.set_xlabel('Return (%)')
ax3.set_ylabel('Frequency')

# Chart 4: Cumulative Returns
ax4 = axes[1, 1]
cumulative_returns = np.cumprod(1 + np.array(trade_log)) - 1 if len(trade_log) > 0 else [0]
ax4.plot(cumulative_returns * 100)
ax4.set_title('Cumulative Trade Returns')
ax4.set_xlabel('Trade #')
ax4.set_ylabel('Cumulative Return (%)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/backtest_analysis.png', dpi=150)
print("✅ Visualization Saved: models/backtest_analysis.png")

# Feature Importance Chart
plt.figure(figsize=(10, 8))
importance_df = pd.DataFrame({
    'feature': final_features,
    'importance': classifier_model.feature_importances_
}).sort_values('importance', ascending=True).tail(20)

plt.barh(importance_df['feature'], importance_df['importance'])
plt.title('Top 20 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150)
print("✅ Feature Importance Saved: models/feature_importance.png")

# -----------------------------------------------------------------------------
# 6. FINAL REPORT
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("🚀 BACKTEST RESULTS")
print("="*70)

print(f"\n📈 PERFORMANCE METRICS:")
print(f"  Initial Capital:    ${INITIAL_CAPITAL:,.2f}")
print(f"  Final Capital:      ${equity_curve[-1]:,.2f}")
print(f"  Total Return:       {total_return:+.2f}%")
print(f"  Annualized Return:  {annualized_return:+.2f}%")

print(f"\n📊 RISK METRICS:")
print(f"  Sharpe Ratio:       {sharpe_ratio:.2f}")
print(f"  Max Drawdown:       {max_drawdown:.2f}%")
print(f"  Calmar Ratio:       {calmar_ratio:.2f}")

print(f"\n🎯 TRADE STATISTICS:")
print(f"  Total Trades:       {total_trades}")
print(f"  Win Rate:           {win_rate:.1f}%")
print(f"  Profit Factor:      {profit_factor:.2f}")
print(f"  Avg Win:            {avg_win:+.2f}%")
print(f"  Avg Loss:           {avg_loss:+.2f}%")

print(f"\n📋 LONG vs SHORT:")
print(f"  Long Trades:        {len(long_trades)} (Win Rate: {long_wr:.1f}%)")
print(f"  Short Trades:       {len(short_trades)} (Win Rate: {short_wr:.1f}%)")

# Quality Assessment
print("\n" + "-"*70)
if sharpe_ratio > 1.0 and max_drawdown > -20 and win_rate > 50:
    print("✅ STRATEGY STATUS: PRODUCTION READY")
elif sharpe_ratio > 0.5 and max_drawdown > -30:
    print("⚠️  STRATEGY STATUS: ACCEPTABLE (needs monitoring)")
else:
    print("❌ STRATEGY STATUS: NEEDS IMPROVEMENT")