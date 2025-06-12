import streamlit as st
import pandas as pd
from datetime import date, datetime
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
from alpha_vantage.timeseries import TimeSeries
import time

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")
YOUR_ALPHA_VANTAGE_API_KEY = 'LUSW8UH3YJV0PK6V'  

st.title("VANSH S&P 500 STOCK FORECASTER")
st.caption("uses data given from Alpha Vantage")
st.caption("not 100% accurate")

tickers_df = pd.read_csv('sp500_tickers.csv')
stocks = tuple(tickers_df['Symbol'])

selected_stocks = st.selectbox("Select dataset for forecast", stocks)

years = st.slider("Years of Forecast", 1, 4)
days = years * 365

@st.cache_data
def load_data(ticker):
    tries = 3
    ts = TimeSeries(key=YOUR_ALPHA_VANTAGE_API_KEY, output_format='pandas')
    for i in range(tries):
        try:
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            if not data.empty:
                
                data = data.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'
                })
                data['Date'] = pd.to_datetime(data.index)
                data.reset_index(drop=True, inplace=True)
                
                data = data[(data['Date'] >= pd.to_datetime(start)) & (data['Date'] <= pd.to_datetime(end))]
                data = data.sort_values(by='Date')
                return data
        except Exception as e:
            st.warning(f"Attempt {i+1}/{tries} failed: {e}")
            time.sleep(2)
    return pd.DataFrame()  


data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")


if data.empty or data['close'].dropna().shape[0] < 2:
    st.error(f"No sufficient data available for {selected_stocks}. Please select a different stock.")
    st.stop()


st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'], y=data["open"], name='stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'], y=data["close"], name='stock_close'))
    figure.layout.update(title="Time series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

plot_raw_data()

df_train = data[['Date', 'close']]
df_train = df_train.rename(columns={"Date": "ds", "close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

st.subheader("Predicted Data")
st.write(forecast.tail())

x = plot_plotly(model, forecast)
st.plotly_chart(x)
st.write("Forecast components")

fig2 = model.plot_components(forecast)
st.write(fig2)

