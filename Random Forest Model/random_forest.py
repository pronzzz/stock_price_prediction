import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Load and clean the data
data = pd.read_csv('TATACONSUM.NS.csv', parse_dates=['Date']).dropna()
features = data.drop(['Date', 'Adj Close'], axis=1)
target = data['Adj Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)
print(f'Accuracy score: {score}')

# Make predictions on the test data
predictions = model.predict(X_test)

# Plot actual and predicted stock prices
plt.figure(figsize=(30,30))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.show()