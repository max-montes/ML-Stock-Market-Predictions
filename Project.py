import yfinance as yf

msft = yf.Ticker("MSFT")

# get all stock info
print(msft.info)