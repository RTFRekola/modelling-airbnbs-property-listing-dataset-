#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Tue 21 Nov 2023 at 11:55 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

import joblib
import json
import numpy as np
import os
import pandas as pd
import typing

from io import StringIO

import inspect
from inspect import signature

#import sklearn
from sklearn import datasets, model_selection, preprocessing, utils
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tabular_data import load_airbnb


# ==============================================
# ===   Functions   ============================
# ==============================================

def custom_tune_regression_model_hyperparameters(model_class, training_set, validation_set, test_set, hyperparameters):

    '''
    This function performs a grid search over a range of hyperparameter values. 

    Variables: 
    - hyperparameters = set of hyperparameters to test, input with function call
    - model_class = the machine learning model class, input with function call
    - training_set = the data training set, input with function call
    - validation_set = the data validation set, input with function call
    - test_set = the data test set, input with function call

    - filtered_par = a dictionary of hyperparameter names and values
    - grid = parameter grid of hyperparameters
    - model = machine learning model initiated with certain hyperparameters
    - param = variable going through all values of the grid in a for loop
    - performance_metrics_list_values = list of all parameter combinations tested
    - performance_metrics_list_RMSE_validation = list of all RMSE validation values
    - RMSE_validation = root mean squared error for the validation set
    - RMSE_validation_0 = comparision value for RMSE_validation
    - y_hat_validation = model prediction

    - best_hyperparameter_values = the values of hyperparameters that produce best model
    - best_model = model with the highest RMSE validation value, returned
    - performance_metrics = results of each test for different combinations of hyperparameters
    '''

    RMSE_validation = 0
    best_model = ""
    best_hyperparameter_values = ""
    performance_metrics = ""
    performance_metrics_list_values = []
    performance_metrics_list_RMSE_validation = []
    grid = list(ParameterGrid(hyperparameters))

    for param in grid:
        # Process hyperparameters
        filtered_par = {key: value for key, value in param.items() 
                        if key in [parameter.name for parameter in 
                        inspect.signature(model_class).parameters.values()]}

        # Initiate the model
        model = model_class(**filtered_par) 

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

    Variables: 
    - model = machine learning model to be tested, input with function call
    - hyperparameters = set of hyperparameters to test, input with function call

    - clf = an object of GridSearchCV initiated with given model and hyperparameters

    - best_estimator = model with the best result in the GridSearchCV testing, returned
    - best_params = the combination of hyperparameters that produces the best result, returned
    - best_score = the value of the best result from GridSearchCV, returned
    '''

    clf = GridSearchCV(model, hyperparameters) #, scoring='accuracy')
    clf.fit(X_train, y_train)
    best_estimator = clf.best_estimator_
    best_score = clf.best_score_
    best_params = clf.best_params_
    metrics_dict = {"validation_score": best_score}
    
    return best_estimator, best_params, metrics_dict
# end tune_regression_model_hyperparameters


def save_model(folder, best_model, best_hyperparameter_values, performance_metrics):

    '''
    This function writes model, hyperparameters and performance metrics into files. 

    Variables: 
    - best_hyperparameter_values = best hyperparameter values, input with function call
    - best_model = best model, input with function call
    - folder = the path to the folder where files should be written, input with function call
    - performance_matrics = performance metrics, input with function call
    '''

    file_name = os.path.join(folder, 'model.joblib')
    joblib.dump(best_model, file_name)
    file_name = os.path.join(folder, 'hyperparameters.json')
    with open(file_name, 'w') as open_file:
        json.dump(best_hyperparameter_values, open_file)
    file_name = os.path.join(folder, 'metrics.json')
    with open(file_name, 'w') as open_file:
        json.dump(performance_metrics, open_file)
# end save_model


def evaluate_sgdregressor():

    '''
    This function evaluates how good the SGDRegressor model is with the Airbnb data.

    Variables: 
    - cv_score = cross validation score
    - score_test = testing set score
    - score_train = training set score
    - sgdr = SGDRegressor object
    - X, y, X_train, y_train, X_test, y_test, y_hat_train, y_hat_test
        = machine learning division of input data into training and testing sets & predictions
    '''

    # Establish model object and make predictions
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
# end evaluate_sgdregressor


def evaluate_all_models(task_folder):

    '''
    This function finds the best set of hyperparameters for a series of regression models (Decision 
    Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor) or classification 
    models (Decision Tree Classifier, Random Forest Classifier, and Gradient Boosting Classifier) 
    running them through either the function tune_regression_model_hyperparameters or 
    tune_classification_model_hyperparameters. These hyperparameters are then used to find the 
    metrics and values of best hyperparameters for each model and save these on the disk. 

    Variables: 
    - task_folder = indicator of whether regression or classification models are being evaluated; 
       also setting the folder into which their results are written, input with function call

    - best_estimator = model with the best hyperparameter values
    - best_params = hyperparameter values that produce the best score
    - best_score = score acquired with the best hyperparameter values
    - folder = path to the location where results are saved into various files
    - hyperparameters = selection of hyperparameter values to be tested and evaluated
    - model = machine learning model (regressor)
    '''

    if (task_folder == "regression"):
        # Decision Tree Regressor
        print("Decision Tree Regressor")
        model = DecisionTreeRegressor()
        hyperparameters = {
            "max_depth": [2, 5, 10],
            "criterion": ["squared_error", "friedman_mse", "poisson"],
            "min_samples_leaf": [2, 1]
        }
        best_estimator, best_params, best_score = tune_regression_model_hyperparameters(model, 
                                                                                    hyperparameters)
    elif (task_folder == "classification"):
        # Decision Tree Classifier
        print("Decision Tree Classifier")
        model = DecisionTreeClassifier()
        hyperparameters = {
            "max_depth": [2, 5, 10],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_leaf": [2, 1]
        }
        best_estimator, best_params, best_score = tune_classification_model_hyperparameters(model, 
                                                                                    hyperparameters)
    # end if
    print("Best estimator = ", best_estimator)
    print("Best hyperparameter values = ", best_params)
    print("Best score = ", best_score)

    folder = os.path.join("models", task_folder, "decision_tree")
    save_model(folder, best_estimator, best_params, best_score)

    if  (task_folder == "regression"):
        # Random Forest Regressor
        print("Random Forest Regressor")
        model = RandomForestRegressor()
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200],
            "criterion": ["squared_error", "friedman_mse", "poisson"],
            "min_samples_leaf": [2, 1]
        }
        best_estimator, best_params, best_score = tune_regression_model_hyperparameters(model, 
                                                                                    hyperparameters)
    elif (task_folder == "classification"):
        # Random Forest Classifier
        print("Random Forest Classifier")
        model = RandomForestClassifier()
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_leaf": [2, 1]
        }
        best_estimator, best_params, best_score = tune_classification_model_hyperparameters(model, 
                                                                                    hyperparameters)
    # end if
    print("Best estimator = ", best_estimator)
    print("Best score = ", best_score)
    print("Best hyperparameter values = ", best_params)

    folder = os.path.join("models", task_folder, "random_forest")
    save_model(folder, best_estimator, best_params, best_score)

    if (task_folder == "regression"):
        # Gradient Boosting Regressor
        print("Gradient Boosting Regressor")
        model = GradientBoostingRegressor()
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200],
            "loss": ["squared_error", "absolute_error", "huber"],
            "min_samples_leaf": [2, 1]
        }
        best_estimator, best_params, best_score = tune_regression_model_hyperparameters(model, 
                                                                                    hyperparameters)
    elif (task_folder == "classification"):
        # Gradient Boosting Classifier
        print("Gradient Boosting Classifier")
        model = GradientBoostingClassifier()
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200],
            "criterion": ["friedman_mse", "squared_error"],
            "min_samples_leaf": [2, 1]
        }
        best_estimator, best_params, best_score = tune_classification_model_hyperparameters(model, 
                                                                                    hyperparameters)
    # end if
    print("Best estimator = ", best_estimator)
    print("Best score = ", best_score)
    print("Best hyperparameter values = ", best_params)

    folder = os.path.join("models", task_folder, "gradient_boosting")
    save_model(folder, best_estimator, best_params, best_score)
# end evaluate_all_models


def find_best_sgdregressor():

    '''
    This function feeds a grid of hyperparameters to regression model tuning function and finds 
    the best set of hyperparameters, i.e. the ones that result in the best score for the model. 

    Variables: 
    - best_hyperparameter_values = the values of hyperparameters that produce best model
    - features_labels_tuple = tuple of airbnb data as a Pandas dataframe and column headers as a list
    - hyperparameters = a selection of ranges of hyperparameters for testing & finding the best values
    - model = variable to hold a chosen machine learning model
    - performance_metrics = results of each test for different combinations of hyperparameters
    - test_set = auxiliary variable to hold X_test
    - training_set = auxiliary variable to hold X_train
    - validation_set = auxiliary variable to hold X_validation

    - best_estimator = model with the best result in the GridSearchCV testing, returned
    - best_params = the combination of hyperparameters that produces the best result, returned
    - best_score = the value of the best result from GridSearchCV, returned
    '''

    # Feed the model class and parameters to a function for finding the best model
    training_set = X_train
    validation_set = X_validation
    test_set = X_test

    hyperparameters = {
        "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
        "learning_rate": ['constant'],  # constant for eta0; invscaling for power_t
        "eta0": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "power_t": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }

    best_model, best_hyperparameter_values, performance_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, training_set, validation_set, test_set, hyperparameters)

    print("Best model = ", best_model)
    print("Best hyperparameter values = ", best_hyperparameter_values)
    print("Performance metrics = ", performance_metrics)

    # Feed the model and parameters to a function to perform SK-learn GridSearchCV
    model = SGDRegressor()
    best_estimator, best_score, best_params = tune_regression_model_hyperparameters(model,
                                                                                    hyperparameters)
    print("Best estimator: " + str(best_estimator))
    print("Best score: " + str(best_score))
    print("Best parameters: " + str(best_params))

    folder = os.path.join("models", "regression", "linear_regression")
    save_model(folder, best_estimator, best_hyperparameter_values, performance_metrics)
# end find_best_sgdregressor


def find_best_model(task_folder):

    '''
    This function compares the machine learning models that have been processed by functions 
    tune_regression_model_hyperparameters and tune_classification_model_hyperparameters. 

    Variables: 
    - task_folder = indicator of whether regression or classification models are being evaluated; 
       also setting the folder into which their results are written, input with function call

    - chosen_model = position of the best model in the list "model_to_compare"
    - contents = contents of a file being read
    - file_name = file to be read with its folder path
    - i = loop parameter
    - metrics_value = value of the metrics of the best machine learning model
    - metrics_value_test = value of the metrics being read from a file
    - model_to_compare = list of machine learning models to be compared
    - one_of_files = loop parameter
    - score_file = list of machine learning models with folder paths
    - score_path = folder path of the best machine learning model

    - hyperparameter_dictionary = hyperparameters of the best model, returned
    - loaded_model = the best model, returned
    - metrics_dictionary = metrics of the best model, returned
    '''

    model_to_compare = []
    model_to_compare.append("decision_tree")
    model_to_compare.append("random_forest")
    model_to_compare.append("gradient_boosting")
    score_file = []
    for i in range(len(model_to_compare)):
        score_file.append(os.path.join("models", task_folder, model_to_compare[i], "metrics.json"))
    # end for
    metrics_value = 0.0
    for one_of_files in range(len(score_file)):
        with open(score_file[one_of_files], 'r') as contents:
            metrics_value_test = float(json.loads(contents.read()).get('validation_accuracy'))
        # end with
        if (metrics_value_test > metrics_value):
            metrics_value = metrics_value_test
            chosen_model = one_of_files
        # end if
    # end for
    score_path = os.path.join("models", task_folder, model_to_compare[chosen_model])
    file_name = os.path.join(score_path, 'model.joblib')
    loaded_model = joblib.load(file_name)
    file_name = os.path.join(score_path, 'hyperparameters.json')
    with open(file_name, 'r') as load_file:
        hyperparameter_dictionary = json.loads(load_file.read())
    file_name = os.path.join(score_path, 'metrics.json')
    with open(file_name, 'r') as load_file:
        metrics_dictionary = json.loads(load_file.read())
    return loaded_model, hyperparameter_dictionary, metrics_dictionary
# end find_best_model


def train_and_evaluate_logistic_regression():

    '''
    This function trains the logistic regression model to predict the "Category" column 
    from the tabular data. Then it calculates various performance values for the training 
    and test sets. 

    Variables: 
    - model = initiated logistic regression 
    - y_hat_train, y_hat_test = machine learning predictions
    '''

    # Train logistic regression to predict the category from the tabular data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    # Compute the F1 score, precision score, recall score and accuracy score
    print("F1 score for train set:  ", f1_score(y_train, y_hat_train, average="macro"))
    print("F1 score for test set:   ", f1_score(y_test, y_hat_test, average="macro"))
    print("Precision for train set: ", precision_score(y_train, y_hat_train, average="macro"))
    print("Precision for test set:  ", precision_score(y_test, y_hat_test, average="macro"))
    print("Recall for train set:    ", recall_score(y_train, y_hat_train, average="macro"))
    print("Recall for test set:     ", recall_score(y_test, y_hat_test, average="macro"))
    print("Accuracy for train set:  ", accuracy_score(y_train, y_hat_train))
    print("Accuracy for test set:   ", accuracy_score(y_test, y_hat_test))

    # Tune hyperparameters
    hyperparameters = {
        "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
        "max_iter": [10, 50, 100, 200],
        "C": [0.5, 1, 2]
    }
    best_model, best_params, metrics_dict = tune_classification_model_hyperparameters(model,
                                                                                      hyperparameters)
    
    print("Best model = ", best_model)
    print("Best hyperparameter values = ", best_params)
    print("Accuracy = ", metrics_dict)
# end train_and_evaluate_logistic_regression


def tune_classification_model_hyperparameters(model, hyperparameters):

    '''
    This function uses SKLearn's GridSearchCV to perform a grid search to find the best set of
    hyperparameters. Then it saves the performance values into files. 

    Variables: 
    - model = machine learning model to be tested, input with function call
    - hyperparameters = set of hyperparameters to test, input with function call

    - clf = an object of GridSearchCV initiated with given model and hyperparameters

    - best_model = model with the best result in the GridSearchCV testing, returned
    - best_params = the combination of hyperparameters that produces the best result, returned
    - metrics_dict = the accuracy of the best result from GridSearchCV, returned
    '''

    # Grid search for the best model
    clf = GridSearchCV(model, hyperparameters, scoring='accuracy')
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    best_params = clf.best_params_
    metrics_dict = {"validation_accuracy": clf.best_score_}

    # Save the model performance values
    folder = os.path.join("models", "classification", "logistic_regression")
    save_model(folder, best_model, best_params, metrics_dict)

    return best_model, best_params, metrics_dict
# end tune_classification_model_hyperparameters



# ==============================================
# ===   Main programme   =======================
# ==============================================

'''
Go through the steps to produce final results and call functions as needed.

Parameters:
- dataset = neural network dataset
- df_in = Pandas dataframe with Airbnb data in it
- df = Pandas dataframe, where "Category" values have been converted into numerical values
- model = chosen model for neural network
- test_loader = dataloader for the test set
- test_set = test set for neural network
- train_len = desired fraction of the length of dataset for training set divisions
- train_loader = dataloader for the training set
- train_set = training set for neural network
- validation_set = validation set for the neural network
- X, y, X_train, y_train, X_test, y_test, X_validation, y_validation
   = machine learning division of input data into training, testing and validation sets
'''

# Load the input data
features_labels_tuple = load_airbnb()

# Using sklearn to train a linear regression model to predict first 
# the "Price_Night" and then the "Category" feature from the tabular data.
df_in = features_labels_tuple[0]
df = df_in.replace(['Treehouses', 'Category', 'Chalets', 'Amazing pools', 'Offbeat', 'Beachfront'],
                   [1, 1, 2, 3, 4, 5])

X = df.iloc[:, 1:]
y = df.iloc[:, 0]
print(y)
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

if __name__ == "__main__":
    # Evaluate SGDregressor model
    evaluate_sgdregressor()
    find_best_sgdregressor()

    # Evaluate a set of alternative regression models
    evaluate_all_models("regression")

    # Find the best regression model from those processed earlier
    loaded_model, hyperparameter_dictionary, metrics_dictionary = find_best_model("regression")
    print("loaded model = ", loaded_model)
    print("hyperparameter dictionary = ", hyperparameter_dictionary)
    print("metrics dictionary = ", metrics_dictionary)

    # Train and evaluate logistic regression
    train_and_evaluate_logistic_regression()

    # Evaluate a set of alternative classification models
    evaluate_all_models("classification")

    # Find the best classification model from those processed earlier
    loaded_model, hyperparameter_dictionary, metrics_dictionary = find_best_model("classification")
    print("loaded model = ", loaded_model)
    print("hyperparameter dictionary = ", hyperparameter_dictionary)
    print("metrics dictionary = ", metrics_dictionary)

# end if
