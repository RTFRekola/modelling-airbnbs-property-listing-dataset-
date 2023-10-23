#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Tue 21 Oct 2023 at 16:44 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

from tabular_data import load_airbnb
import numpy as np
import pandas as pd
import inspect
from inspect import signature
import sklearn
from sklearn import datasets, model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


# ==============================================
# ===   Functions   ============================
# ==============================================

def custom_tune_regression_model_hyperparameters(model_class, training_set, validation_set, test_set, hyperparameters):

    '''
    This function performs a grid search over a range of hyperparameter
    values. 
    '''

    RMSE_validation = 0
    best_model = ""
    best_hyperparameter_values = ""
    performance_metrics = ""
    performance_metrics_list_values = []
    performance_metrics_list_RMSE_validation = []
    grid = list(ParameterGrid(hyperparameters))
    #print(grid)

    for param in grid:
        # Process hyperparameters
        filtered_par = {key: value for key, value in param.items() 
                        if key in [parameter.name for parameter in 
                        inspect.signature(model_class).parameters.values()]}
        #filtered_par = {key: value for key, value in hyperparameters.items() 
        #                if key in [parameter.name for parameter in 
        #                inspect.signature(SGDRegressor).parameters.values()]}
        # Initiate the model
        model = model_class(**filtered_par) 
        #print(filtered_par)

        # Fit the model
        model.fit(X_train,y_train)
        # Make predictions
        y_hat_validation = model.predict(X_validation)
        # Evaluate predictions
        RMSE_validation_0 = mean_squared_error(y_validation, 
                                               y_hat_validation, 
                                               squared=False)
        # Store results of the above for the return
        if (RMSE_validation_0 > RMSE_validation):
            best_model = model
            RMSE_validation = RMSE_validation_0
            best_hyperparameter_values = param
        # end if
        performance_metrics_list_values.append(param)
        performance_metrics_list_RMSE_validation.append(RMSE_validation_0)
    # end for
    performance_metrics = {
        'validation_RMSE': performance_metrics_list_RMSE_validation, 
        'hyperparameter_values': performance_metrics_list_values
    }

    return best_model, best_hyperparameter_values, performance_metrics
    # end custom_tune_regression_model_hyperparameters


def tune_regression_model_hyperparameters(model, hyperparameters):

    '''
    This function uses SKLearn's GridSearchCV to perform a grid search.
    '''

    clf = GridSearchCV(model, hyperparameters)
    clf.fit(X_train, y_train)
    best_estimator = clf.best_estimator_
    best_score = clf.best_score_
    best_params = clf.best_params_
    
    '''
    grid = GridSearchCV(model, hyperparameters)
    # Fit the model for grid search
    grid.fit(X_train, y_train)
    # Print best parameter after tunin
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
    # Print classification report
    print(classification_report(y_test, grid_predictions))
    '''
    return best_estimator, best_score, best_params
    # end tune_regression_model_hyperparameters



# ==============================================
# ===   Main programme   =======================
# ==============================================

# Load the input data
features_labels_tuple = load_airbnb()

# Using sklearn to train a linear regression model to predict 
# the "Price_Night feature from the tabular data
df = features_labels_tuple[0]
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X = scale(X)
y = scale(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

sgdr = SGDRegressor()

sgdr.fit(X_train, y_train)
y_hat_train = sgdr.predict(X_train)
y_hat_test = sgdr.predict(X_test)

# Compute the key measures of performance for the regression model
cv_score = cross_val_score(sgdr, X, y, cv = 10)
print("CV mean score: ", cv_score.mean())
print()

print("MSE (train):", mean_squared_error(y_train, y_hat_train))
print("MSE (test):", mean_squared_error(y_test, y_hat_test))
print()

print("RMSE (train):", mean_squared_error(y_train, y_hat_train, squared=False))
print("RMSE (test):", mean_squared_error(y_test, y_hat_test, squared=False))
print()

score_train = sgdr.score(X_train, y_train)
print("R-squared (train):", score_train)
score_test = sgdr.score(X_test, y_test)
print("R-squared (test):", score_test)
print()

# Feed the model class and parameters to a function for finding the best model
training_set = X_train
validation_set = X_validation
test_set = X_test
'''
hyperparameters = {
    "max_samples": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "n_estimators": [2, 4, 8, 16, 32, 64, 1024]
}
'''
hyperparameters = {
    "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    "max_iter": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
    "learning_rate": ['constant'],  # constant for eta0; invscaling for power_t
    "eta0": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    "power_t": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

# best_model, best_hyperparameter_values, performance_metrics = custom_tune_regression_model_hyperparameters(model_class, training_set, validation_set, test_set, hyperparameters)
#best_model, best_hyperparameter_values, performance_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, training_set, validation_set, test_set, hyperparameters)

#print("Best model = ", best_model)
#print("Best hyperparameter values = ", best_hyperparameter_values)
#print("Performance metrics = ", performance_metrics)

'''
best_model, best_hyperparameter_values, performance_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, training_set, validation_set, test_set, hyperparameters)

print("Best model = ", best_model)
print("Best hyperparameter values = ", best_hyperparameter_values)
print("Performance metrics = ", performance_metrics)
'''

# Feed the model and parameters to a function to perform SK-learn GridSearchCV
model = SGDRegressor()
best_estimator, best_score, best_params = tune_regression_model_hyperparameters(model, hyperparameters)
print("Best estimator: " + str(best_estimator))
print("Best score: " + str(best_score))
print("Best parameters: " + str(best_params))
