import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import timedelta


# -----------------get data from yahoo for a given period------------------------------------
@st.cache(allow_output_mutation=True)
def get_stock_data(ticker_symbol, start_date, end_date):
    try:
        df = yf.download(ticker_symbol, start_date + timedelta(days=1), end_date, progress=False)
        df.index = pd.to_datetime(df.index).date
        return df
    except ValueError:
        print("invalid input")


# ----------------- Get ticker data from a file downloaded from nasdaq -------------------------
@st.cache(allow_output_mutation=True)
def get_tickers_data():
    try:
        df = pd.DataFrame(pd.read_csv('nasdaq_screener_full.csv')).fillna("Unknown")
        return df
    except ValueError:
        print("Not able to load stocks ticker Data")


# ----------------- Get ticker symbol for a given stock name ------------------------------------
def get_ticker_symbol(companies, stock):
    return companies.query('Name == @stock').index[0]


# ----------------- Get Descriptive analysis data for the stock ------------------------------------
def get_descriptive_analytics(stock_df):
    descriptive_data = round((stock_df.describe()), 2)
    descriptive_data = descriptive_data.T
    descriptive_data['cov'] = round(descriptive_data['std']/descriptive_data['mean'], 2)
    print(descriptive_data)
    return descriptive_data.T
