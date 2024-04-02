import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


end_date = datetime.now()
start_date = end_date - timedelta(days=21)

msft_data = yf.download("MSFT", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))


msft_data_last_11 = msft_data.tail(11)

# SMA and Momentum 10day
sma_10_day = msft_data_last_11['Close'][:-1].mean()

momentum_10_day = msft_data_last_11['Close'].iloc[-1] - msft_data_last_11['Close'].iloc[0]

# stochastic oscillator K%
lowest_low = msft_data_last_11['Low'][:-1].min()
highest_high = msft_data_last_11['High'][:-1].max()
current_close = msft_data_last_11['Close'].iloc[-1]
k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100

momentum_10_day,k_percent,sma_10_day
