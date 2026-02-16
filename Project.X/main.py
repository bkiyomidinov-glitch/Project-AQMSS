import pandas as pd
import numpy as np
from datetime import datetime
import os

# Optional AI predictor (Market Environment Classifier)
try:
    from predictor import load_predictor
except Exception:
    load_predictor = None

# -------------------------------
# Конфигурируемые параметры
# -------------------------------
weights = {
    "volatility_regime": 0.25,  # Current ATR vs historical mean (was: liquidity)
    "true_liquidity": 0.15,     # Volume-based metric
    "structure": 0.30,
    "momentum": 0.20,
    "volatility_level": 0.10    # ATR std dev (volatility stability)
}
# Percentile-based thresholds (calculated from data distribution)
thresholds = {
    "NO_TRADE": 4,      # Bottom 25%
    "WEAK": 6,          # 25-50%
    "NORMAL": 7.5,      # 50-75%
    "HIGH": 10          # Top 25%
}
# Note: Thresholds should be backtested and adjusted based on win rate
structure_window = 20    # количество свечей для structure
momentum_window = 5      # количество свечей для momentum
atr_window = 50          # ATR для расчёта volatility regime и volatility level
liquidity_window = 10    # Volume window for true liquidity
time_weighted = True     # учитывать веса последних свечей

# -------------------------------
# Чтение данных
# -------------------------------
try:
    df = pd.read_csv(r'C:\Users\bkiyo\Downloads\EURUSD_60_2025-01-20_2026-01-19.csv', header=None)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'volume2', 'other']
    # Skip the first row (corrupted header)
    df = df.iloc[1:].reset_index(drop=True)
except Exception as e:
    print("Error reading CSV:", e)
    exit()

# Валидация данных
if len(df) < 50:
    print("Error: Not enough data. Require at least 50 candles.")
    exit()

df = df.dropna()

# Конвертирование цен в float
df['open'] = pd.to_numeric(df['open'], errors='coerce')
df['high'] = pd.to_numeric(df['high'], errors='coerce')
df['low'] = pd.to_numeric(df['low'], errors='coerce')
# Convert volume columns
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
df['volume2'] = pd.to_numeric(df['volume2'], errors='coerce')
df = df.dropna()

# Validate data integrity
if (df['high'] < df['low']).any():
    print("Warning: Found candles where high < low. Removing corrupted data.")
    df = df[df['high'] >= df['low']]

if len(df) < 50:
    print("Error: Not enough valid data after integrity check.")
    exit()

# -------------------------------
# ATR для волатильности
# -------------------------------
df['HL'] = df['high'] - df['low']
df['HC'] = abs(df['high'] - df['close'].shift(1))
df['LC'] = abs(df['low'] - df['close'].shift(1))
df['TR'] = df[['HL','HC','LC']].max(axis=1)
df['ATR'] = df['TR'].rolling(atr_window).mean()

# Calculate statistics BEFORE dropping NaN to use full dataset
mean_atr = df['ATR'].mean()
atr_std = df['ATR'].std()

# Now drop NaN
df = df.dropna()

# Verify we have enough data after cleanup
required_rows = max(structure_window, momentum_window)
if len(df) < required_rows:
    print(f"Error: Not enough data after processing. Have {len(df)}, need {required_rows}")
    exit()

# Use the last valid ATR
current_atr = df['ATR'].iloc[-1]

# -------------------------------
# Volatility Regime Score (Current ATR vs Historical Mean)
# This measures if volatility is above/below average (NOT true liquidity)
# -------------------------------
atr_ratio = current_atr / mean_atr if mean_atr > 0 else 0
volatility_regime_raw = atr_ratio * 10
volatility_regime_score = np.clip(volatility_regime_raw, 0, 10)

# -------------------------------
# True Liquidity Score (Volume-Based Metric)
# Higher volume = better liquidity for entry/exit
# -------------------------------
recent_volume = df['volume'].iloc[-liquidity_window:].values
mean_volume = df['volume'].mean()
std_volume = df['volume'].std()

if mean_volume > 0:
    # Calculate volume z-score for recent candles
    volume_ratio = recent_volume.mean() / mean_volume if mean_volume > 0 else 0
    liquidity_raw = np.clip(volume_ratio * 10, 0, 10)
    true_liquidity_score = np.clip(liquidity_raw, 0, 10)
else:
    true_liquidity_score = 5  # Neutral if no volume data

# -------------------------------
# Volatility Level Score (ATR Standard Deviation)
# Measures stability: high std = unstable, low std = stable
# -------------------------------
volatility_level_raw = (atr_std / mean_atr * 10) if mean_atr > 0 else 0
volatility_level_score = np.clip(volatility_level_raw, 0, 10)

# -------------------------------
# Structure Score (линейная регрессия High/Low normalized by ATR)
# -------------------------------
highs = df['high'].iloc[-structure_window:].values
lows = df['low'].iloc[-structure_window:].values
x = np.arange(structure_window)
slope_high = np.polyfit(x, highs, 1)[0]
slope_low = np.polyfit(x, lows, 1)[0]
mean_candle_range = (highs - lows).mean()
# Normalize slope by average candle range
structure_raw = ((abs(slope_high) + abs(slope_low)) / mean_candle_range) * 10 if mean_candle_range > 0 else 0
structure_score = np.clip(structure_raw, 0, 10)

# -------------------------------
# Momentum Score (учёт диапазона свечи)
# -------------------------------
momentum_range = (df['high'].iloc[-momentum_window:] - df['low'].iloc[-momentum_window:]).values
momentum_atr = df['ATR'].iloc[-momentum_window:].values

# Check for zero/NaN values in momentum_atr
if np.any(momentum_atr == 0) or np.any(np.isnan(momentum_atr)):
    momentum_score = 0
else:
    if time_weighted:
        weights_array = np.linspace(0.5, 1.0, momentum_window)
        # Calculate weighted average of range/ATR ratios
        ratio_weighted = (momentum_range / momentum_atr) * weights_array
        momentum_raw = (ratio_weighted.sum() / weights_array.sum()) * 10
    else:
        momentum_raw = (momentum_range / momentum_atr).mean() * 10
    momentum_score = np.clip(momentum_raw, 0, 10)

# -------------------------------
# TOTAL SCORE
# -------------------------------
scores = {
    "volatility_regime": volatility_regime_score,
    "true_liquidity": true_liquidity_score,
    "structure": structure_score,
    "momentum": momentum_score,
    "volatility_level": volatility_level_score
}

# Validate scores - replace NaN with 0
for k, v in scores.items():
    if np.isnan(v) or np.isinf(v):
        print(f"Warning: {k} score is invalid ({v}), setting to 0")
        scores[k] = 0

total_score = sum(scores[k]*weights[k] for k in scores)

# Ensure total_score is valid
if np.isnan(total_score) or np.isinf(total_score):
    print("Error: Total score is invalid. Check data quality.")
    total_score = 0

# -------------------------------
# Результат
# -------------------------------
if total_score < thresholds['NO_TRADE']:
    market_condition = "NO TRADE"
elif total_score < thresholds['WEAK']:
    market_condition = "WEAK CONDITIONS"
elif total_score < thresholds['NORMAL']:
    market_condition = "NORMAL MARKET"
else:
    market_condition = "HIGH QUALITY MARKET"

# -------------------------------
# Вывод
# -------------------------------
print("=== Market Quality Score (AMQSS v2 - Fixed) ===")
print("\n📊 Scores per factor:")
print(f"  Volatility Regime (ATR ratio): {volatility_regime_score:.2f}/10")
print(f"  True Liquidity (Volume-based): {true_liquidity_score:.2f}/10")
print(f"  Structure (Trend strength): {structure_score:.2f}/10")
print(f"  Momentum (Candle range): {momentum_score:.2f}/10")
print(f"  Volatility Level (Stability): {volatility_level_score:.2f}/10")
print(f"\n💰 TOTAL SCORE: {total_score:.2f}/10")
print(f"📈 Market Condition: {market_condition}")
print("\n⚠️  Note: Thresholds should be backtested for your strategy")
print("\n📋 Last 10 candles:")
print(df[['time', 'open', 'high', 'low', 'close', 'volume', 'ATR']].tail(10).to_string(index=False))
# -------------------------------
# Save results to CSV for dashboard
# -------------------------------
results_dir = r'C:\Users\bkiyo\Desktop\Project.X\results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create results dataframe
# Use timestamp from ~30 bars before end (so forward performance can be computed)
# This ensures the timestamp has enough future bars for ML training
safe_idx = max(0, len(df) - 30)
amqss_timestamp = pd.to_datetime(df['time'].iloc[safe_idx])

results_df = pd.DataFrame({
    'timestamp': [amqss_timestamp],
    'total_score': [total_score],
    'volatility_regime': [volatility_regime_score],
    'true_liquidity': [true_liquidity_score],
    'structure': [structure_score],
    'momentum': [momentum_score],
    'volatility_level': [volatility_level_score],
    'market_condition': [market_condition],
    'current_atr': [current_atr],
    'mean_atr': [mean_atr],
    'atr_std': [atr_std],
    'recent_volume': [df['volume'].iloc[-liquidity_window:].mean() if 'volume' in df.columns else 0]
})

# -------------------------------
# AI Integration: Market Environment Classifier
# Loads trained model if available. Classifies market quality (not price direction).
# Returns None if model not trained yet.
# -------------------------------
ai_high_quality_prob = None
models_dir = r'C:\Users\bkiyo\Desktop\Project.X\models'

current_features = {
    'volatility_regime': float(volatility_regime_score),
    'true_liquidity': float(true_liquidity_score),
    'structure': float(structure_score),
    'momentum': float(momentum_score),
    'volatility_level': float(volatility_level_score),
    'current_atr': float(current_atr),
    'recent_volume': float(df['volume'].iloc[-liquidity_window:].mean() if 'volume' in df.columns else 0)
}

if load_predictor is not None:
    try:
        predictor = load_predictor(models_dir)
        ai_high_quality_prob = predictor.predict(current_features)
        print(f"\n🤖 AI Market Environment Score: {ai_high_quality_prob:.4f}")
    except Exception as e:
        print(f"\n⚠️  AI Predictor unavailable: {e}")

results_df['ai_high_quality_prob'] = [ai_high_quality_prob]

# Save to CSV (append mode with proper header handling)
csv_path = os.path.join(results_dir, 'market_scores.csv')
if os.path.exists(csv_path):
    # Read existing data, append new row, write back
    try:
        existing_df = pd.read_csv(csv_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    except Exception as e:
        print(f"Warning: Could not append to existing CSV ({e}). Overwriting.")
    results_df.to_csv(csv_path, index=False)
else:
    results_df.to_csv(csv_path, index=False)

print(f"\n💾 Results saved to: {csv_path}")