import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import quandl

data = quandl.get("NSE/TATAGLOBAL")

data.head(10)

plt.figure(figsize=(16,8))
plt.plot(data['Close'], label='Closing Price')

"""Classification Problem : Buy(+1) or sell(-1) the stock"""

# @title Default title text
data['Open - Close']= data['Open'] - data['Close']
data['High - Low']= data['High'] - data['Low']
data = data.dropna()

"""Input Features to predict whether customer should buy or sell the stock"""

x = data[['Open - Close', 'High - Low']]
x.head()

"""Intention is to store +1 for the buy signal and -1 fro the sell signal. The target is 'Y' for classification task."""

Y = np.where(data['Close'].shift(-1)>data['Close'],1,-1)

Y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,Y,test_size=0.25, random_state = 44)

"""Implementation of KNN Classifier"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#using gridseaarch to find the parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

#fit the model
model.fit(x_train, y_train)

#Accuracy Score
accuracy_train = accuracy_score(y_train, model.predict(x_train))
accuracy_test = accuracy_score(y_test, model.predict(x_test))

print('Train_data Accuracy: %.2f' %accuracy_train)
print('Test_data Accuracy: %.2f' %accuracy_test)

prediction_classification = model.predict(x_test)

actual_predicted_data = pd.DataFrame({'Actual Class':y_test, 'Predicted Class':prediction_classification})

actual_predicted_data.head(10)

"""Regression Problem: KNN"""

y = data['Close']

y

"""Implementation of KNN Regression"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y, test_size=0.25, random_state=44)

#using gridseaarch to find the parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)

#fit the model
model_reg.fit(x_train_reg, y_train_reg)
predictions = model_reg.predict(x_train_reg)

print(predictions)

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
rms

valid = pd.DataFrame({'Actual Close':y_test_reg, 'Predicted CLose Value':predictions})

valid.head(10)