import yfinance as yf
import pandas as pd
import numpy as np

def fetch_benchmark_data(ticker, years):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].dropna()

def calculate_benchmark_returns(price_data):
    returns = price_data.pct_change().fillna(0)
    return returns

def get_data(ticker, years):
    prices = fetch_benchmark_data(ticker, years)
    returns = calculate_benchmark_returns(prices)
    return prices, returns