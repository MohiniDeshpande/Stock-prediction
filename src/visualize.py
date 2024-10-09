# src/visualize.py

import matplotlib.pyplot as plt
from datetime import timedelta

def visualize_stock_trend(df, symbol, days):
    today = df['Date'].max()
    past_days = today - timedelta(days=days)
    df_timeframe = df[df['Date'] >= past_days]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_timeframe['Date'], df_timeframe['Open'], marker='o', label='Open', color='blue')
    plt.plot(df_timeframe['Date'], df_timeframe['High'], marker='o', label='High', color='orange')
    plt.plot(df_timeframe['Date'], df_timeframe['Low'], marker='o', label='Low', color='green')
    plt.plot(df_timeframe['Date'], df_timeframe['Close'], marker='o', label='Close', color='red')

    plt.title(f"Stock Price Data for {symbol} (Last {days} Days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(loc="best")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
