 ## Tata Global Stock Price Prediction Using KNN Implementation

### Dataset Source: Quandl

### Main Targets:

- Regression Problem: Predict the Close Target.
- Classification Problem: Predict whether to BUY or SELL the stock.

  - BUY: +1
  - SELL: -1

### Approach: K-Nearest Neighbors using GridSearchCV

---

This code offers a comprehensive demonstration of the K-Nearest Neighbors algorithm for machine learning tasks. It covers both classification and regression problems, making it an invaluable resource for those who want to explore KNN's capabilities.

#### Libraries:

The code imports essential libraries such as Pandas, NumPy, Matplotlib, and Quandl.

#### Stock Data:

Real-world stock data is fetched from Quandl. Data preprocessing is performed before splitting the data into training and testing sets.

#### Classification Task:

- Features are extracted and the target variable, which represents whether to buy or sell the stock, is prepared.
- K-Nearest Neighbors classifier is applied with grid search cross-validation to optimize hyperparameters.
- Model performance is evaluated on training and test data.

#### Regression Task:

- Closing stock prices are predicted using K-Nearest Neighbor regression.
- Grid search cross-validation is employed to determine optimal hyperparameters.
- Root mean squared error (RMSE) is calculated as a metric to assess the model's performance.

#### Outputs:

- Actual and predicted values are presented for both classification and regression tasks.
- This provides a clear demonstration of KNN algorithm's application in real-world scenarios.﻿
