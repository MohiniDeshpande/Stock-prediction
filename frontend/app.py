# frontend/app.py

from src.data import get_stock_data, scrape_google_news
from src.sentiment import analyze_sentiment
from src.model import prepare_stock_data, predict_stock_movement
from src.visualize import visualize_stock_trend

def main():
    symbol = input("Enter the stock symbol you want to track (e.g., AAPL): ").upper()
    days = int(input("Enter the time frame in days for the stock graph: "))
    
    stock_data = get_stock_data(symbol)
    news_headlines = scrape_google_news(symbol)
    sentiments, avg_sentiment_score = analyze_sentiment(news_headlines)
    df = prepare_stock_data(stock_data, avg_sentiment_score)

    visualize_stock_trend(df, symbol, days)
    prediction, model = predict_stock_movement(df)
    trend = "upwards" if prediction == 1 else "downwards"
    print(f"\nFinal Prediction: The stock is expected to trend {trend}.")

if __name__ == "__main__":
    main()
