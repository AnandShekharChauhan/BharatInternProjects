import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


start_date = '2020-01-01'
end_date = '2021-12-31'

df = yf.download(stock_symbol, start=start_date, end=end_date)

data = df['Adj Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)


train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

sequence_length = 10
X_train = create_sequences(train_data, sequence_length)
y_train = train_data[sequence_length:]
X_test = create_sequences(test_data, sequence_length)
y_test = test_data[sequence_length:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=64)


predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size+sequence_length:], test_data, label='True Prices')
plt.plot(df.index[train_size+sequence_length:], predicted_prices, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
