import yfinance as yf

# Get ticket
msft = yf.Ticker("MSFT")

# Get data
hist = msft.history(period="max")
curr = msft.info

# Get components for calculation of Larry Williams Percent Range
highest_high = hist['High'].max()
lowest_low = hist['Low'].min()
current_close = curr['regularMarketPreviousClose']

# Calculate Larry Williams Percent Range
williams_percent_range = ((highest_high - current_close) / (highest_high - lowest_low)) * -100