import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta

# set to Apple for now
ticker = yf.Ticker('AAPL')

expiration_dates = ticker.options
print(expiration_dates)

pd.set_option('display.max_columns', None)
chain = ticker.option_chain(expiration_dates[0])
print(chain.calls)
print(chain.puts)
