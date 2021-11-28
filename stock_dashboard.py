import stocks_data as sd
from moving_averages import compute_moving_averages
import streamlit as st
from predictive_analysis import linear_reg
from datetime import date

# ---- Get companies data and put it in the stock selection box ---- #
companies = sd.get_tickers_data()
companies = companies.set_index('Symbol')
stock_list = sorted(set(companies.loc[:, "Name"]))
stock_name = st.sidebar.selectbox("Stock name", options=stock_list)
ticker_symbol = sd.get_ticker_symbol(companies, stock_name)

# ---- Create start date and End date input box and set defaults ---- #
date_today = date.today()
date_year_back = date_today.replace(year=date_today.year - 1, month=date_today.month, day=date_today.day)
start_date = st.sidebar.date_input('Start date', date_year_back)
end_date = st.sidebar.date_input('End date', date_today)

# ---- Display error in case user choose incorrect date range ---- #
if start_date > end_date:
    st.sidebar.error('Error: End date should not be before start date.')

# --- Print company details for selected stock --- #
st.header(f'* {ticker_symbol} - {stock_name}*')

# --- Get historical data for selected stock for given date range --- #
stock_history = sd.get_stock_data(ticker_symbol, start_date, end_date)

# --- Display historical data in a table--- #
st.subheader('Historical Data')
with st.expander(f'View historical data for {ticker_symbol} from {start_date} to {end_date}', expanded=False):
    st.table(stock_history)

# --- Display historical data in a table--- #
st.subheader('Descriptive Analysis')
with st.expander(f'View Descriptive analysis for {ticker_symbol} from {start_date} to {end_date}', expanded=False):
    st.table(sd.get_descriptive_analytics(stock_history))

# --- Display Moving Averages data on a Graph--- #
st.subheader('Moving Averages')
with st.expander(f'Visualize Moving Averages for {ticker_symbol}', expanded=False):
    col1, col2, col3 = st.columns(3)
    # --- input start data and end date for Moving average computation --- #
    moving_avg_start_date = col1.date_input('Choose Start date', start_date)
    moving_avg_end_date = col2.date_input('Choose End date', end_date)
    moving_avg_window = col3.slider(label='Window for Moving Average', min_value=2, max_value=200, value=20, step=1)
    if moving_avg_start_date > moving_avg_end_date:
        st.write('Error: End date should not be before start date.')
    else:
        # --- Fetch Stock history data again if user changes inputs--- #
        if start_date != moving_avg_start_date or end_date != moving_avg_end_date:
            stock_data = sd.get_stock_data(ticker_symbol, moving_avg_start_date, moving_avg_end_date)
            moving_avg = compute_moving_averages(stock_data, 'Adj Close', moving_avg_window)
        else:
            moving_avg = compute_moving_averages(stock_history, 'Adj Close', moving_avg_window)
        st.line_chart(moving_avg)

# --- Display Predictive analysis--- #
st.subheader('Predictive Analysis')
with st.expander(f'View Predictive analysis for {ticker_symbol} based on data from {start_date} to {end_date}',
                 expanded=False):
    col1, col2, col3 = st.columns(3)
    training_start_date = col1.date_input('Training Model Start date', start_date)
    training_end_date = col2.date_input('Training Model End date', end_date)
    predictive_days = col3.slider(label='Input Days for prediction', min_value=1, max_value=100, value=5, step=1)
    if moving_avg_start_date > moving_avg_end_date:
        st.write('Error: End date should not be before start date.')
    else:
        # --- Fetch Stock history data again if user changes inputs--- #
        if start_date != training_start_date or end_date != training_end_date:
            stock_data_for_training = sd.get_stock_data(ticker_symbol, training_start_date, training_end_date)
            fig = linear_reg(stock_data_for_training, predictive_days, stock_name)
        else:
            fig = linear_reg(stock_history, predictive_days, stock_name)
        if not fig is None:
            st.pyplot(fig)
