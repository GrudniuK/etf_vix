
#.venv\scripts\activate
#instalowanie pakietow:
#python -m pip install pandas
#pip install ta
#   https://github.com/bukosabino/ta
#   https://technical-analysis-library-in-python.readthedocs.io/en/latest/
#pip install yfinance
# https://github.com/ranaroussi/yfinance

#deactivate
#pip freeze


# pozyskanie danych OHCL:
#https://blog.quantinsti.com/historical-market-data-python-api/

#%%
import yfinance as yf

#################################33

#msft = yf.Ticker("MSFT")
#hist = msft.history(period="max")
#hist
data = yf.download("SPY", start="2017-01-01", end="2017-04-30")
data



#%%
a = 1
print(a)

# %%

msg = 'Hello world!'
print(msg)
msg.capitalize

# %%
import numpy as np