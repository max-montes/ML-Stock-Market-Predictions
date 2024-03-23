import yfinance as yf

msft = yf.Ticker("MSFT")

# get all stock info
print(msft.info)

stochastic_k = 1 #Taken from Bottom3Attributes.py
