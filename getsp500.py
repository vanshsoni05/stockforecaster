import requests
import pandas as pd

# URL to fetch S&P 500 companies' data
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Fetch the data from the URL
response = requests.get(url)
tables = pd.read_html(response.text)

# The first table on the page contains the S&P 500 companies
sp500_table = tables[0]

# Extract the ticker symbols
tickers = sp500_table['Symbol']

# Save to a CSV file
tickers.to_csv('sp500_tickers.csv', index=False, header=True)


