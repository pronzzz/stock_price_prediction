import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the Dataset
data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/TATACONSUM.NS?period1=820454400&period2=1706400000&interval=1d&events=history&includeAdjustedClose=true')

# Data Analysing and Data Modelling
data.dropna(inplace=True)
data.set_index('Date', inplace=True)
data.drop('Adj Close', axis=1, inplace=True)
X = data.drop('Close', axis=1)
y = data['Close']
corr = data.corr()

# Model Creation and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model
score = model.score(X_test, y_test)
print('The Accuracy score of the model is:', score)

# Make Predictions and display Results
y_pred = model.predict(X_test)
prediction_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.figure(figsize=(50,50))
prediction_results.plot(title='Actual vs. Predicted Stock Prices')
plt.show()