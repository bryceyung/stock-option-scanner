import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
sp500_table = tables[0]

# sp500_tickers = sp500_table['Symbol'].tolist()
# print(sp500_tickers)

try:
    with open('Stocks_to_buy_List_symbols.csv', 'r') as file:
        sp500_tickers = [line.strip().split(',')[0] for line in file if line.strip()]
except FileNotFoundError:
    print("CSV file not found.")
    sp500_tickers = []


end = date.today()
start = end - relativedelta(years=5)

def bollinger_bands(df, window=20):
    df[f'SMA{window}'] = df['Close'].rolling(window=window).mean()
    df[f'STD{window}'] = df['Close'].rolling(window=window).std()
    df[f'Upper_Band{window}'] = df[f'SMA{window}'] + (df[f'STD{window}'] * 2)
    df[f'Lower_Band{window}'] = df[f'SMA{window}'] - (df[f'STD{window}'] * 2)

    return df

def create_rsi(df,window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta>0, 0))
    loss = (-delta.where(delta<0, 0))

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi_value = 100 - (100/(1+rs))

    df['RSI'] = rsi_value

    return df

def money_flow_index(df,window=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    window=14
    positive_mf_sum = positive_flow.rolling(window-window).sum()
    negative_mf_sum = negative_flow.rolling(window-window).sum()
    money_flow_ratio = positive_mf_sum / negative_mf_sum
    mfi = 100 - (100 / (1+money_flow_ratio))
    df['MFI'] = mfi
    return df

# def algorithm(ticker, start=start, end=end, interval='1d'):
#     # Download data using yfinance
#     df = yfinance.download(ticker, start=start, end=end, interval=interval)
#     df.columns = [col[0] for col in df.columns]

#     # Apply technical indicators
#     df = bollinger_bands(df=df, window=12)
#     df = bollinger_bands(df=df, window=20)
#     df = bollinger_bands(df=df, window=50)
#     df = create_rsi(df=df, window=14)
#     df = money_flow_index(df=df, window=14)

#     # Extract necessary columns
#     low = df['Low']
#     high = df['High']
#     sma12 = df['SMA12']
#     sma20 = df['SMA20']
#     sma50 = df['SMA50']
#     bollinger_upper_20 = df['Upper_Band20']
#     bollinger_lower_20 = df['Lower_Band20']
#     bollinger_upper_50 = df['Upper_Band50']
#     bollinger_lower_50 = df['Lower_Band50']
#     rsi = df['RSI']
#     mfi = df['MFI']

#     # Initialize signals
#     buy_signal = []
#     sell_signal = []
#     action = []

#     # Generate buy/sell signals
#     for i in range(len(df)):
#         low_val = low[i]
#         high_val = high[i]
#         sma12_val = sma12[i]
#         sma20_val = sma20[i]
#         sma50_val = sma50[i]
#         bollinger_upper_20_val = bollinger_upper_20[i]
#         bollinger_lower_20_val = bollinger_lower_20[i]
#         bollinger_upper_50_val = bollinger_upper_50[i]
#         bollinger_lower_50_val = bollinger_lower_50[i]
#         rsi_val = rsi[i]
#         mfi_val = mfi[i]

#         # Define buy and sell conditions
#         buy_condition = (low_val < bollinger_lower_20_val) and (rsi_val < 30 and mfi_val < 25)
#         sell_condition = (high_val > bollinger_upper_20_val) and (rsi_val > 70 and mfi_val > 75)

#         if buy_condition:
#             buy_signal.append(low_val * 0.95)
#             sell_signal.append(np.nan)
#             action.append("Buy")
#         elif sell_condition:
#             buy_signal.append(np.nan)
#             sell_signal.append(high_val * 1.05)
#             action.append("Sell")
#         else:
#             buy_signal.append(np.nan)
#             sell_signal.append(np.nan)
#             action.append("Hold")

#     # Assign signals to DataFrame
#     df['buy_signal'] = buy_signal
#     df['sell_signal'] = sell_signal
#     df['action'] = action

#     # Initialize portfolio values
#     initial_cash = 10000
#     cash = initial_cash
#     position = 0
#     portfolio_value = []

#     # Calculate Buy and Hold Strategy Value
#     share_cost = df['Open'].iloc[0]
#     num_shares = initial_cash / share_cost
#     df['Buy_Hold_Value'] = df['Close'] * num_shares

#     # Calculate Portfolio Value based on signals
#     for i in range(len(df)):
#         current_action = df['action'].iloc[i]
#         price = df['Close'].iloc[i]

#         if current_action == 'Buy' and cash > 0:
#             position = cash / price
#             cash = 0
#         elif current_action == 'Sell' and position > 0:
#             cash = position * price
#             position = 0
        
#         # Portfolio value at each step
#         portfolio_value.append(cash + position * price)

#     # Assign Portfolio Value to DataFrame
#     df['Portfolio_Value'] = portfolio_value

#     return df

def stochastic_rsi(df, window=10, smooth_k=4, smooth_d=4):
    delta = df['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    stoch_rsi = (rsi - rsi.rolling(window).min()) / (rsi.rolling(window).max() - rsi.rolling(window).min())
    df['Stoch_RSI'] = stoch_rsi

    df['%K'] = stoch_rsi.rolling(smooth_k).mean()
    df['%D'] = df['%K'].rolling(smooth_d).mean()

    return df

def algorithm(ticker, start=start, end=end, interval='1d'):
    df = yf.download(ticker, start=start, end=end, interval=interval)

    if df.empty:
        print(f"No data for {ticker}")
        return pd.DataFrame()

    # Calculate Indicators
    df = bollinger_bands(df, window=20)
    df = create_rsi(df, window=14)
    df = money_flow_index(df, window=14)

    # Calculate Stochastic RSI
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Signal Calculation
    df['Signal'] = np.where((df['%K'].shift(1) < df['%D'].shift(1)) & (df['%K'] > df['%D']), 'Buy',
                     np.where((df['%K'].shift(1) > df['%D'].shift(1)) & (df['%K'] < df['%D']), 'Sell', 'Hold'))

    # Buy and Hold Calculation
    initial_cash = 10000
    df['Buy_Hold_Value'] = (initial_cash / df['Open'].iloc[0]) * df['Close']

    return df

# Run the algorithm on all S&P 500 tickers
result_dict = {}

for ticker in sp500_tickers:
    try:
        print(f">> Now Running: {ticker}")
        df = algorithm(ticker, start=start, end=end)
        if not df.empty:
            latest_signal = df['Signal'].iloc[-1]
            result_dict[ticker] = latest_signal
    except Exception as e:
        print(f"Error with {ticker}: {e}")

# Display results
sp500_results = pd.DataFrame.from_dict(result_dict, orient='index', columns=['Signal'])
sp500_results = sp500_results[sp500_results['Signal'].isin(['Buy', 'Sell'])]
sp500_results.sort_values(by='Signal', inplace=True)

# Save results to CSV (more readable format)
sp500_results.reset_index(inplace=True)
sp500_results.columns = ['Ticker', 'Signal']
sp500_results.to_csv('sp500_signals.csv', index=False)

print(sp500_results.head())