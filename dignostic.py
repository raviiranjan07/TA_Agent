"""
Data Diagnostics - Find what's wrong with the features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("DATA DIAGNOSTICS")
print("="*70)

# Load data
df = pd.read_csv('data/btc_with_features.csv', index_col='time', parse_dates=True)
print(f"\n✅ Loaded {len(df):,} rows with {len(df.columns)} columns")

# =============================================================================
# CHECK 1: Price Data Quality
# =============================================================================
print("\n" + "="*70)
print("CHECK 1: Price Data Quality")
print("="*70)

print("\n📊 Price Statistics:")
print(df[['close', 'volume']].describe())

print("\n📊 Sample of raw data (first 10 rows):")
print(df[['close', 'high', 'low', 'volume']].head(10))

print("\n📊 Sample of raw data (last 10 rows):")
print(df[['close', 'high', 'low', 'volume']].tail(10))

# Check for constant prices
price_changes = df['close'].diff().dropna()
constant_periods = (price_changes == 0).sum()
print(f"\n⚠️  Periods with no price change: {constant_periods:,} ({constant_periods/len(df)*100:.2f}%)")

# Check price volatility
print(f"\n📈 Price Range:")
print(f"  Min:  ${df['close'].min():,.2f}")
print(f"  Max:  ${df['close'].max():,.2f}")
print(f"  Std:  ${df['close'].std():,.2f}")

# =============================================================================
# CHECK 2: Moving Averages
# =============================================================================
print("\n" + "="*70)
print("CHECK 2: Moving Averages Correlation")
print("="*70)

ma_cols = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200']
available_mas = [col for col in ma_cols if col in df.columns]

if available_mas:
    # Calculate correlation
    ma_corr = df[available_mas].corr()
    print("\n📊 Moving Average Correlations:")
    print(ma_corr.round(4))
    
    # Check for perfect correlations
    perfect_corr = (ma_corr == 1.0).sum().sum() - len(available_mas)  # Subtract diagonal
    print(f"\n⚠️  Perfect correlations (1.000) found: {perfect_corr}")
    
    if perfect_corr > 0:
        print("\n🚨 This is ABNORMAL! Moving averages should not be perfectly correlated!")
        print("   Possible causes:")
        print("   1. All prices are identical (flat data)")
        print("   2. Feature calculation error")
        print("   3. Data replication issue")

# =============================================================================
# CHECK 3: Feature Value Ranges
# =============================================================================
print("\n" + "="*70)
print("CHECK 3: Feature Value Ranges")
print("="*70)

# Check key features
key_features = ['rsi_14', 'volatility_50', 'volume_ratio', 'returns', 'obv_sma']
for feat in key_features:
    if feat in df.columns:
        print(f"\n{feat}:")
        print(f"  Min:    {df[feat].min():.6f}")
        print(f"  Max:    {df[feat].max():.6f}")
        print(f"  Mean:   {df[feat].mean():.6f}")
        print(f"  Std:    {df[feat].std():.6f}")
        print(f"  NaN:    {df[feat].isna().sum():,}")
        
        # Check for constant values
        unique_vals = df[feat].nunique()
        if unique_vals < 10:
            print(f"  ⚠️  WARNING: Only {unique_vals} unique values!")

# =============================================================================
# CHECK 4: Time Period Coverage
# =============================================================================
print("\n" + "="*70)
print("CHECK 4: Time Period Coverage")
print("="*70)

print(f"\n📅 Date Range:")
print(f"  Start: {df.index.min()}")
print(f"  End:   {df.index.max()}")
print(f"  Days:  {(df.index.max() - df.index.min()).days}")

# Check time gaps
time_diffs = df.index.to_series().diff()
mode_diff = time_diffs.mode()[0]
print(f"\n⏱️  Time Resolution: {mode_diff}")

gaps = time_diffs[time_diffs > mode_diff * 2]
print(f"\n⚠️  Large time gaps found: {len(gaps)}")
if len(gaps) > 0:
    print(f"   Largest gap: {gaps.max()}")

# =============================================================================
# CHECK 5: Feature Distribution
# =============================================================================
print("\n" + "="*70)
print("CHECK 5: Feature Distribution Analysis")
print("="*70)

# Exclude non-feature columns
exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'num_trades', 
                'typical_price', 'obv', 'ad_line', 'tr']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Check for features with zero variance
zero_var_features = []
for col in feature_cols:
    if col in df.columns:
        var = df[col].var()
        if pd.notna(var) and var < 1e-10:
            zero_var_features.append(col)

if zero_var_features:
    print(f"\n🚨 Features with ZERO variance (constant values):")
    for feat in zero_var_features:
        print(f"   • {feat}")
else:
    print("\n✅ All features have variance")

# =============================================================================
# CHECK 6: Target Variable Quality
# =============================================================================
print("\n" + "="*70)
print("CHECK 6: Target Variable Quality")
print("="*70)

# Create targets
forecast_horizon = 15
df['price_change_pct'] = (df['close'].shift(-forecast_horizon) - df['close']) / df['close'] * 100
df['direction'] = (df['price_change_pct'] > 0).astype(int)

print(f"\n📊 Target Statistics:")
print(f"  Price change % - Mean: {df['price_change_pct'].mean():.4f}%")
print(f"  Price change % - Std:  {df['price_change_pct'].std():.4f}%")
print(f"  Price change % - Min:  {df['price_change_pct'].min():.4f}%")
print(f"  Price change % - Max:  {df['price_change_pct'].max():.4f}%")

print(f"\n📊 Direction Distribution:")
print(df['direction'].value_counts(normalize=True))

# Check if target is predictable
if df['price_change_pct'].std() < 0.01:
    print(f"\n🚨 WARNING: Target has very low variance!")
    print(f"   This means prices barely change - model can't learn!")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "="*70)
print("Creating Diagnostic Visualizations...")
print("="*70)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# 1. Price over time
axes[0, 0].plot(df.index, df['close'], linewidth=0.5)
axes[0, 0].set_title('Price Over Time')
axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Price changes distribution
axes[0, 1].hist(df['price_change_pct'].dropna(), bins=100, alpha=0.7)
axes[0, 1].set_title('Price Changes Distribution')
axes[0, 1].set_xlabel('Price Change (%)')
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].grid(True, alpha=0.3)

# 3. Moving averages
if all(col in df.columns for col in ['sma_20', 'sma_50', 'sma_200']):
    sample = df.iloc[-5000:]  # Last 5000 points
    axes[1, 0].plot(sample.index, sample['close'], label='Close', linewidth=1)
    axes[1, 0].plot(sample.index, sample['sma_20'], label='SMA 20', linewidth=1)
    axes[1, 0].plot(sample.index, sample['sma_50'], label='SMA 50', linewidth=1)
    axes[1, 0].plot(sample.index, sample['sma_200'], label='SMA 200', linewidth=1)
    axes[1, 0].set_title('Moving Averages (Last 5000 points)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# 4. RSI
if 'rsi_14' in df.columns:
    axes[1, 1].plot(df.index, df['rsi_14'], linewidth=0.5)
    axes[1, 1].axhline(70, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(30, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('RSI')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)

# 5. Volume
axes[2, 0].bar(df.index, df['volume'], width=0.0001, alpha=0.5)
axes[2, 0].set_title('Volume Over Time')
axes[2, 0].set_ylabel('Volume')
axes[2, 0].grid(True, alpha=0.3)

# 6. Feature correlation heatmap (sample)
sample_features = ['sma_20', 'sma_50', 'sma_100', 'sma_200', 'rsi_14', 'volatility_50']
available_sample = [f for f in sample_features if f in df.columns]
if len(available_sample) > 2:
    corr = df[available_sample].corr()
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                ax=axes[2, 1], cbar_kws={'shrink': 0.8})
    axes[2, 1].set_title('Feature Correlations')

plt.tight_layout()
plt.savefig('viz_data_diagnostics.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: viz_data_diagnostics.png")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE!")
print("="*70)
print("\nPlease review the output above and the visualization.")
print("Look for:")
print("  • Constant or near-constant prices")
print("  • Perfect (1.000) correlations between features")
print("  • Features with zero variance")
print("  • Abnormal value ranges")
print("="*70)