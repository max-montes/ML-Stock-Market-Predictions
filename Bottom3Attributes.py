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