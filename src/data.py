# src/data.py

import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

def get_stock_data(symbol):
    api_key = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

def scrape_google_news(symbol):
    query = f"{symbol} stock news"
    url = f'https://www.google.com/search?q={query}&tbm=nws'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    headlines = []
    soup = BeautifulSoup(response.text, 'html.parser')
    for item in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
        headline = item.get_text()
        headlines.append(headline)
    return headlines[:5]
