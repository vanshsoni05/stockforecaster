
import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import pandas_datareader as data

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d") 

st.title("VANSH S&P 500 STOCK FORECASTER")
st.caption("uses data given from yahoo finance")
tickers_df = pd.read_csv('sp500_tickers.csv')
stocks = tuple(tickers_df['Symbol'])


selected_stocks = st.selectbox("Select dataset for forecast", stocks)

years = st.slider("Years of Forecast", 1, 4)
days = years * 365


@st.cache_data
def load_data(ticker):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start, end=end)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"Attempt {attempt+1} - Error fetching data: {e}")
            time.sleep(5)  # Wait before retrying
    return None
data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
if data is not None:
    data_load_state.text("Loading data...done!")
else:
    data_load_state.text("Failed to load data.")
### NEW CODE FOR ERRORS#####

######################################

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'],y=data["Open"],name= 'stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'],y=data["Close"],name= 'stock_close'))
    figure.layout.update(title = "Time series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(figure)


plot_raw_data()

#forecasting

df_train = data[['Date','Close']]
df_train = df_train.rename(columns = {"Date": "ds","Close": "y"})

model = Prophet()

model.fit(df_train)
future = model.make_future_dataframe(periods = days)
forecast = model.predict(future)

st.subheader("Predicted Data")
st.write(forecast.tail())

x = plot_plotly(model,forecast)
st.plotly_chart(x)
st.write("Forecast components")

fig2 = model.plot_components(forecast)
st.write(fig2)
