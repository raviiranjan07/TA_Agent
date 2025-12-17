# """
# Step 2: Data Visualization
# Now that we know data is clean, let's visualize it!
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Set style
# sns.set_style("darkgrid")
# plt.rcParams['figure.figsize'] = (15, 10)

# # Load the data we saved
# print("Loading data...")
# df = pd.read_csv('data/btc_sample_3days.csv')
# df['time'] = pd.to_datetime(df['time'])
# df.set_index('time', inplace=True)

# print(f"✅ Loaded {len(df):,} candles")
# print(f"Date range: {df.index.min()} to {df.index.max()}\n")

# # =============================================================================
# # VISUALIZATION 1: Price Overview
# # =============================================================================
# print("Creating Visualization 1: Price Overview...")

# fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# # 1. Closing Price
# axes[0].plot(df.index, df['close'], linewidth=1, color='black', label='Close Price')
# axes[0].fill_between(df.index, df['low'], df['high'], alpha=0.2, color='blue', label='High-Low Range')
# axes[0].set_title('BTC Price - 3 Days (1-minute candles)', fontsize=16, fontweight='bold')
# axes[0].set_ylabel('Price (USD)', fontsize=12)
# axes[0].legend(loc='upper left')
# axes[0].grid(True, alpha=0.3)

# # Add price annotations
# max_price = df['close'].max()
# min_price = df['close'].min()
# max_time = df['close'].idxmax()
# min_time = df['close'].idxmin()

# axes[0].annotate(f'High: ${max_price:,.0f}', 
#                 xy=(max_time, max_price), 
#                 xytext=(10, 20), 
#                 textcoords='offset points',
#                 bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.7),
#                 arrowprops=dict(arrowstyle='->', color='green'))

# axes[0].annotate(f'Low: ${min_price:,.0f}', 
#                 xy=(min_time, min_price), 
#                 xytext=(10, -30), 
#                 textcoords='offset points',
#                 bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.7),
#                 arrowprops=dict(arrowstyle='->', color='red'))

# # 2. Volume
# axes[1].bar(df.index, df['volume'], width=0.0007, alpha=0.6, color='steelblue', label='Volume')
# axes[1].set_title('Trading Volume', fontsize=16, fontweight='bold')
# axes[1].set_ylabel('Volume (BTC)', fontsize=12)
# axes[1].legend(loc='upper left')
# axes[1].grid(True, alpha=0.3)

# # Highlight high volume periods
# high_volume_threshold = df['volume'].quantile(0.95)
# high_volume = df[df['volume'] > high_volume_threshold]
# axes[1].scatter(high_volume.index, high_volume['volume'], 
#                color='red', s=30, alpha=0.7, 
#                label=f'High Volume (>95th percentile)')
# axes[1].legend(loc='upper left')

# # 3. Number of Trades
# axes[2].plot(df.index, df['num_trades'], linewidth=1, color='purple', alpha=0.7)
# axes[2].set_title('Number of Trades per Minute', fontsize=16, fontweight='bold')
# axes[2].set_ylabel('Number of Trades', fontsize=12)
# axes[2].set_xlabel('Time', fontsize=12)
# axes[2].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('viz1_price_overview.png', dpi=300, bbox_inches='tight')
# # plt.show()
# # print("✅ Saved: viz1_price_overview.png\n")
# plt.close()

# # =============================================================================
# # VISUALIZATION 2: Price Distribution & Patterns
# # =============================================================================
# print("Creating Visualization 2: Price Patterns...")

# fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# # 1. Price Distribution
# axes[0, 0].hist(df['close'], bins=50, alpha=0.7, color='blue', edgecolor='black')
# axes[0, 0].axvline(df['close'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
# axes[0, 0].axvline(df['close'].median(), color='green', linestyle='--', linewidth=2, label='Median')
# axes[0, 0].set_title('Price Distribution', fontsize=14, fontweight='bold')
# axes[0, 0].set_xlabel('Price (USD)')
# axes[0, 0].set_ylabel('Frequency')
# axes[0, 0].legend()
# axes[0, 0].grid(True, alpha=0.3)

# # 2. Price Returns (% change)
# df['returns'] = df['close'].pct_change() * 100
# axes[0, 1].hist(df['returns'].dropna(), bins=100, alpha=0.7, color='green', edgecolor='black')
# axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
# axes[0, 1].set_title('Price Returns Distribution (%)', fontsize=14, fontweight='bold')
# axes[0, 1].set_xlabel('Return (%)')
# axes[0, 1].set_ylabel('Frequency')
# axes[0, 1].grid(True, alpha=0.3)

# # 3. Volume Distribution
# axes[1, 0].hist(df['volume'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
# axes[1, 0].set_title('Volume Distribution', fontsize=14, fontweight='bold')
# axes[1, 0].set_xlabel('Volume (BTC)')
# axes[1, 0].set_ylabel('Frequency')
# axes[1, 0].grid(True, alpha=0.3)

# # 4. Price vs Volume Scatter
# axes[1, 1].scatter(df['volume'], df['returns'].abs(), alpha=0.3, s=10)
# axes[1, 1].set_title('Volume vs Absolute Returns', fontsize=14, fontweight='bold')
# axes[1, 1].set_xlabel('Volume (BTC)')
# axes[1, 1].set_ylabel('Absolute Return (%)')
# axes[1, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('viz2_distributions.png', dpi=300, bbox_inches='tight')
# # plt.show()
# print("✅ Saved: viz2_distributions.png\n")
# plt.close()

# # =============================================================================
# # VISUALIZATION 3: Intraday Patterns
# # =============================================================================
# print("Creating Visualization 3: Time-based Patterns...")

# # Add time features
# df['hour'] = df.index.hour
# df['day'] = df.index.day_name()

# fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# # 1. Average price by hour
# hourly_avg = df.groupby('hour')['close'].mean()
# axes[0, 0].bar(hourly_avg.index, hourly_avg.values, alpha=0.7, color='blue')
# axes[0, 0].set_title('Average Price by Hour (UTC)', fontsize=14, fontweight='bold')
# axes[0, 0].set_xlabel('Hour of Day')
# axes[0, 0].set_ylabel('Average Price (USD)')
# axes[0, 0].grid(True, alpha=0.3)

# # 2. Average volume by hour
# hourly_volume = df.groupby('hour')['volume'].mean()
# axes[0, 1].bar(hourly_volume.index, hourly_volume.values, alpha=0.7, color='steelblue')
# axes[0, 1].set_title('Average Volume by Hour (UTC)', fontsize=14, fontweight='bold')
# axes[0, 1].set_xlabel('Hour of Day')
# axes[0, 1].set_ylabel('Average Volume (BTC)')
# axes[0, 1].grid(True, alpha=0.3)

# # 3. Volatility by hour (std of returns)
# hourly_volatility = df.groupby('hour')['returns'].std()
# axes[1, 0].bar(hourly_volatility.index, hourly_volatility.values, alpha=0.7, color='red')
# axes[1, 0].set_title('Volatility by Hour (Std of Returns)', fontsize=14, fontweight='bold')
# axes[1, 0].set_xlabel('Hour of Day')
# axes[1, 0].set_ylabel('Volatility (Std %)')
# axes[1, 0].grid(True, alpha=0.3)

# # 4. Number of trades by hour
# hourly_trades = df.groupby('hour')['num_trades'].mean()
# axes[1, 1].bar(hourly_trades.index, hourly_trades.values, alpha=0.7, color='purple')
# axes[1, 1].set_title('Average Trades by Hour', fontsize=14, fontweight='bold')
# axes[1, 1].set_xlabel('Hour of Day')
# axes[1, 1].set_ylabel('Average Number of Trades')
# axes[1, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('viz3_time_patterns.png', dpi=300, bbox_inches='tight')
# # plt.show()
# print("✅ Saved: viz3_time_patterns.png\n")
# plt.close()

# # =============================================================================
# # Summary Statistics
# # =============================================================================
# print("="*70)
# print("SUMMARY STATISTICS")
# print("="*70)

# print(f"\n📊 Price Statistics:")
# print(f"   Range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
# print(f"   Mean: ${df['close'].mean():,.2f}")
# print(f"   Volatility (std): ${df['close'].std():,.2f}")

# print(f"\n📈 Returns Statistics:")
# print(f"   Mean return: {df['returns'].mean():.4f}%")
# print(f"   Volatility: {df['returns'].std():.4f}%")
# print(f"   Max gain: {df['returns'].max():.4f}%")
# print(f"   Max loss: {df['returns'].min():.4f}%")

# print(f"\n💰 Volume Statistics:")
# print(f"   Total: {df['volume'].sum():,.2f} BTC")
# print(f"   Average: {df['volume'].mean():,.2f} BTC/min")
# print(f"   Peak hour (UTC): {hourly_volume.idxmax()}:00")

# print(f"\n🔥 Most Active Hour: {hourly_trades.idxmax()}:00 UTC ({hourly_trades.max():,.0f} trades avg)")
# print(f"📉 Least Active Hour: {hourly_trades.idxmin()}:00 UTC ({hourly_trades.min():,.0f} trades avg)")

# print("\n" + "="*70)
# print("✅ VISUALIZATIONS COMPLETE!")
# print("="*70)
# print("\nCreated 3 visualizations:")
# print("  1. viz1_price_overview.png - Price, volume, trades timeline")
# print("  2. viz2_distributions.png - Statistical distributions")
# print("  3. viz3_time_patterns.png - Hourly patterns")
# print("\nNext: Feature engineering!")
# print("="*70)


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/btc_with_features.csv",
    parse_dates=["time"]
)

df = df.sort_values("time")

# plt.figure(figsize=(14, 5))
# plt.plot(df["time"], df["price_range_pct"])
# plt.title("Price Range % Over Time")
# plt.xlabel("Time")
# plt.ylabel("Price Range (%)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["price_range_pct"], bins=100)
plt.title("Distribution of Price Range %")
plt.xlabel("Price Range (%)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
