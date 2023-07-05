import pandas as pd
import ta

df = pd.read_csv('../BTCUSDT_1h.csv')

# Drop the first 10% of rows
df = df.iloc[int(len(df)*0.1):].reset_index(drop=True)

df['gain'] = df['close'] / df['open'] - 1
df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'])
df['volume_ma_14'] = df['volume'].rolling(14).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma_14'].shift(1)
df['size'] = (df['high'] - df['low']) / df['close']

def add_features(df):

  float_cols = ['open', 'high', 'low', 'close', 'volume']
  df[float_cols] = df[float_cols].astype(float)

  df['rsi'] = ta.momentum.stochrsi(df['close'], window=14)

  df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

  df['tsi'] = ta.momentum.TSIIndicator(df['close']).tsi()

  df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

  df['gain'] = df['close'] / df['open'] - 1
  df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'])

  df['close_10'] = df['close'].rolling(10).mean()
  df['close_10_diff'] = df['close'] / df['close_10'] - 1

  df['close_20'] = df['close'].rolling(20).mean()
  df['close_20_diff'] = df['close'] / df['close_20'] - 1

  df['close_50'] = df['close'].rolling(50).mean()
  df['close_50_diff'] = df['close'] / df['close_50'] - 1

  df['volume_20_diff'] = df['volume'] / df['volume'].rolling(20).mean() - 1

  df['bar_size'] = (df['high'] - df['low']) / df['close']

  # True if the next gain is in the top 10% of gains
  df['next_gain_is_big'] = (df['gain'].shift(-1) > df['gain'].shift(-1).quantile(0.9)).astype(int)

  features = ['open', 'high', 'low', 'close', 'gain', 'ibs', 'volume', 'close_10_diff', 'close_20_diff', 'close_50_diff']

  features = ['open', 'high', 'low', 'close', 'gain', 'ibs', 'close_10_diff', 'close_20_diff', 'close_50_diff', 'rsi', 'adx']

  features = ['gain', 'ibs', 'close_10_diff', 'close_20_diff', 'close_50_diff', 'rsi', 'adx', 'bar_size', 'next_gain_is_big']


  return features

features = add_features(df)

# Convert time column to datetime
df['date'] = pd.to_datetime(df['time'], unit='ms')

# Remove NA 
df = df.dropna().reset_index(drop=True)

# Select the features to use
df = df[features + ['date']]

# Save the data
df.to_csv('../BTCUSDT_1h_features.csv', index=False)