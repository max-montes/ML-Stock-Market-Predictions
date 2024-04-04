def get_SMA(data, lookback_period=10):
    return data['Close'].rolling(window=lookback_period).mean()

def get_momentum(data, lookback_period=10):
    return data['Close'] - data['Close'].shift(lookback_period)

def get_stochastic_oscillator_k_percent(data, lookback_period=14):
    lowest_low = data['Low'].rolling(window=lookback_period).min()
    highest_high = data['High'].rolling(window=lookback_period).max()
    return ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

def get_stochastic_oscillator_k_percent_moving_average(data, lookback_period=14):
    lowest_low = data['Low'].rolling(window=lookback_period).min()
    highest_high = data['High'].rolling(window=lookback_period).max()
    k_percent = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

    return k_percent.rolling(window=lookback_period).mean()

def get_relative_strength_index(data, lookback_period=14):
    diff = data['Close'].diff()
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window=lookback_period).mean()
    avg_loss = loss.rolling(window=lookback_period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_moving_average_convergence_divergence(data, lookback_period_short=12, lookback_period_long=26):
    short_ema = data['Close'].ewm(span=lookback_period_short, adjust=False).mean()
    long_ema = data['Close'].ewm(span=lookback_period_long, adjust=False).mean()
    return short_ema - long_ema

def get_williams_percent_range(data, lookback_period=14):
    rolling_highest_high = data['High'].rolling(window=lookback_period).max()
    rolling_lowest_low = data['Low'].rolling(window=lookback_period).min()
    return -100 * ((rolling_highest_high - data['Close']) / (rolling_highest_high - rolling_lowest_low))

def get_a_d_index(data):
    money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    money_flow_volume = data['Volume'] * money_flow_multiplier
    a_d = money_flow_volume.cumsum()
    return a_d

def get_commodity_channel_index(data, lookback_period):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3.0
    moving_average = typical_price.rolling(window=lookback_period).mean()
    mean_deviation = typical_price.rolling(window=lookback_period).apply(
        lambda tp: (tp - tp.mean()).abs().mean()
    )
    return (typical_price - moving_average) / (0.015 * mean_deviation)