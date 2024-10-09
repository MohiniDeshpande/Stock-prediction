# src/model.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=64):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.transformer(x, x)
        x = x[:, -1, :]
        return torch.sigmoid(self.fc(x))

def prepare_stock_data(stock_data, sentiment_score):
    time_series = stock_data.get('Time Series (Daily)', {})
    rows = []
    for date, values in time_series.items():
        rows.append({
            'Date': datetime.strptime(date, "%Y-%m-%d"),
            'Open': float(values['1. open']),
            'High': float(values['2. high']),
            'Low': float(values['3. low']),
            'Close': float(values['4. close']),
            'Volume': float(values['5. volume']),
            'Sentiment_Score': sentiment_score
        })
    df = pd.DataFrame(rows).sort_values(by='Date')
    df['Price_Change'] = df['Close'].shift(-1) - df['Close']
    df['Price_Up'] = (df['Price_Change'] > 0).astype(int)
    df.dropna(inplace=True)
    return df

def predict_stock_movement(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Score']].values
    y = df['Price_Up'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = TransformerModel(input_dim=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_labels = (y_pred.numpy() > 0.5).astype(int)
    
    latest_data = X_scaled[-1].view(1, 1, X_scaled.shape[2])
    prediction = model(latest_data).item()
    return 1 if prediction > 0.5 else 0, model
