# Stock Price Prediction with Sentiment Analysis

## Overview
This project implements a stock price prediction model using sentiment analysis and historical stock data from the NASDAQ market. The model employs FinBERT, a specialized transformer model for financial sentiment analysis, to evaluate the sentiment of recent news headlines related to the target stock. The historical stock data is retrieved via the Alpha Vantage API, and the model predicts whether the stock price will trend upwards or downwards based on sentiment scores and other market indicators.

The predictive model incorporates a Transformer architecture and is trained to classify stock movements. Upon evaluation, the model achieved an accuracy of 61% and an AUC score of 0.60. This indicates that while the model provides a baseline for predicting stock movements, it may require further tuning and optimization for enhanced performance.

## Input Features:

Historical Stock Data: The model incorporates various stock metrics, including Open, High, Low, Close, and Volume, as well as sentiment scores generated from news headlines using FinBERT.

Sentiment Analysis: FinBERT processes recent news headlines to determine sentiment polarity, yielding a sentiment score that reflects the overall market sentiment towards the stock.

1. Transformer Architecture:

- The model consists of several encoder layers that process the input time series data. Each layer employs multi-head self-attention to focus on different parts of the input, enabling the model to learn intricate relationships between features over time.
  
- The output from the encoder layers is passed through a feedforward neural network, which further processes the information to make a final classification.
  
2. Training Process:

- The model is trained using historical data split into training and test sets. The binary cross-entropy loss function is used, as the task is to classify stock movement as either upward or downward.
  
- An Adam optimizer is employed to update the model's parameters during training, ensuring efficient convergence.

## Important Notes
API Key: To run this code, you must obtain an Alpha Vantage API key. Replace the placeholder API key in data.py with your own.

Limitations: Please note that the Alpha Vantage API allows only 25 requests per day for free accounts, making this code unsuitable for large-scale applications.

## How to Run

Clone this repository to your local machine.

Install the required dependencies using:
pip install -r requirements.txt

Run the script by executing:

python main.py


Follow the prompts to enter the stock symbol and time frame for analysis.

![Screenshot 2024-10-09 152507](https://github.com/user-attachments/assets/f0397f03-f87c-45c8-ad0f-f1ae71fb93cf)
