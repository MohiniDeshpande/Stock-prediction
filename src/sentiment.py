# src/sentiment.py

import torch
from transformers import pipeline

def analyze_sentiment(headlines):
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", device=device)
    sentiments = []
    total_score = 0.0
    for headline in headlines:
        result = sentiment_pipeline(headline)
        sentiment_label = result[0]['label']
        sentiment_score = result[0]['score']
        sentiments.append((sentiment_label, sentiment_score))
        total_score += (sentiment_score if sentiment_label == 'positive' else -sentiment_score)
    avg_sentiment_score = total_score / len(sentiments) if sentiments else 0
    return sentiments, avg_sentiment_score
