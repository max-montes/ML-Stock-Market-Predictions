def get_SMA(data, lookback_period=10):
    return data['Close'].rolling(window=lookback_period).mean()

def get_momentum(data, lookback_period=10):
    return data['Close'] - data['Close'].shift(lookback_period)

def get_stochastic_oscillator_k_percent(data, lookback_period=20):
    lowest_low = data['Low'].rolling(window=lookback_period).min()
    highest_high = data['High'].rolling(window=lookback_period).max()
    return ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100