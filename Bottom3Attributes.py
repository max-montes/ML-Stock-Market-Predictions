#import ta
import yfinance as yf

msft = yf.Ticker("MSFT")

hist = msft.history(period="max")
curr = msft.info

def get_williams_percent_range(data, lookback_period):
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

    deviation = abs(typical_price - moving_average)
    mean_deviation = deviation.rolling(window=lookback_period, min_periods=1).mean()

    return (typical_price - moving_average) / (0.015 * mean_deviation)

period = 12

#print(get_williams_percent_range(hist, period))
#print(ta.momentum.williams_r(hist['High'], hist['Low'],  hist['Close'], lbp=period))

#print(get_a_d_index(hist))
#print(ta.volume.acc_dist_index(hist['High'],  hist['Low'],  hist['Close'],  hist['Volume']))

#print(get_commodity_channel_index(hist, period))
#print(ta.trend.cci(hist['High'],  hist['Low'],  hist['Close'], window=period, constant=0.015))

hist['Williams % R'] = get_williams_percent_range(hist, period)
hist['A/D Index'] = get_a_d_index(hist)
hist['CCI'] = get_commodity_channel_index(hist, period)

print(hist)