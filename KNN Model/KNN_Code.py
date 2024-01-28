 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_collection():
    data = quandl.get("NSE/TATAGLOBAL")
    return data

def data_preprocessing(data):
    """
    Performs data preprocessing steps like creating new features and cleaning the data.

    Args:
        data: The data to be preprocessed.

    Returns:
        The preprocessed data.
    """
    # Create new features
    data["open_minus_close"] = data["Open"] - data["Close"]
    data["high_minus_low"] = data["High"] - data["Low"]
    # Drop rows with missing values
    data = data.dropna()
    return data

def feature_extraction(data):
    """
    Extracts features from the data.

    Args:
        data: The data to extract features from.

    Returns:
        The extracted features.
    """
    x = data[["open_minus_close", "high_minus_low"]]
    return x

def target_encoding(data):
    """
    Encodes the target variable into a binary representation.

    Args:
        data: The data to encode the target variable for.

    Returns:
        The encoded target variable.
    """
    y = np.where(data["Close"].shift(-1) > data["Close"], 1, -1)
    return y

def data_splitting(x, y):
    """
    Splits the data into training and testing sets.

    Args:
        x: The features.
        y: The target variable.

    Returns:
        The training and testing sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=44)
    return x_train, x_test, y_train, y_test

def gridsearch_knn(x_train, y_train):
    """
    Performs grid search to find the optimal hyperparameters for the KNN classifier.

    Args:
        x_train: The training features.
        y_train: The training target variable.

    Returns:
        The KNN classifier with the optimal hyperparameters.
    """
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
    knn = neighbors.KNeighborsClassifier()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)
    return model

def training_testing(model, x_train, y_train, x_test, y_test):
    """
    Trains and tests the KNN classifier.

    Args:
        model: The KNN classifier to be trained and tested.
        x_train: The training features.
        y_train: The training target variable.
        x_test: The testing features.
        y_test: The testing target variable.

    Returns:
        The training and testing accuracy scores.
    """
    # Train the model
    model.fit(x_train, y_train)

    # Test the model
    y_pred = model.predict(x_test)

    # Calculate the accuracy scores
    accuracy_train = accuracy_score(y_train, model.predict(x_train))
    accuracy_test = accuracy_score(y_test, y_pred)

    return accuracy_train, accuracy_test

def data_collection_knn_regression():
    data = quandl.get("NSE/TATAGLOBAL")
    return data

def data_preprocessing_knn_regression(data):
    """
    Performs data preprocessing steps for the KNN regression model.

    Args:
        data: The data to be preprocessed.

    Returns:
        The preprocessed data.
    """
    # Create new features
    data["open_minus_close"] = data["Open"] - data["Close"]
    data["high_minus_low"] = data["High"] - data["Low"]
    # Drop rows with missing values
    data = data.dropna()
    # Extract the target variable
    y = data["Close"]
    # Extract the features
    x = data[["open_minus_close", "high_minus_low"]]
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=44)
    return x_train, x_test, y_train, y_test

def gridsearch_knn_regression(x_train, y_train):
    """
    Performs grid search to find the optimal hyperparameters for the KNN regression model.

    Args:
        x_train: The training features.
        y_train: The training target variable.

    Returns:
        The KNN regression model with the optimal hyperparameters.
    """
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
    knn_reg = neighbors.KNeighborsRegressor()
    model_reg = GridSearchCV(knn_reg, params, cv=5)
    model_reg.fit(x_train, y_train)
    return model_reg

def training_testing_regression(model_reg, x_train, y_train, x_test, y_test):
    """
    Trains and tests the KNN regression model.

    Args:
        model_reg: The KNN regression model to be trained and tested.
        x_train: The training features.
        y_train: The training target variable.
        x_test: The testing features.
        y_test: The testing target variable.

    Returns:
        The root mean squared error (RMSE) score.
    """
    # Train the model
    model_reg.fit(x_train, y_train)

    # Test the model
    predictions = model_reg.predict(x_test)

    # Calculate the RMSE score
    rms = np.sqrt(np.mean(np.power((np.array(y_test) - np.array(predictions)), 2)))

    return rms

def main():
    # Classification Problem: Buy(+1) or sell(-1) the stock
    data = data_collection()
    data = data_preprocessing(data)
    x = feature_extraction(data)
    y = target_encoding(data)
    x_train, x_test, y_train, y_test = data_splitting(x, y)
    model = gridsearch_knn(x_train, y_train)
    accuracy_train, accuracy_test = training_testing(model, x_train, y_train, x_test, y_test)
    print("Train data accuracy:", accuracy_train)
    print("Test data accuracy:", accuracy_test)

    # Regression Problem: KNN
    data = data_collection_knn_regression()
    x_train, x_test, y_train, y_test = data_preprocessing_knn_regression(data)
    model_reg = gridsearch_knn_regression(x_train, y_train)
    rms = training_testing_regression(model_reg, x_train, y_train, x_test, y_test)
    print("Root mean squared error:", rms)

if __name__ == "__main__":
    main()﻿
