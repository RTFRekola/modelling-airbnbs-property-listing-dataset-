#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Sun 4 Feb 2024 at 16:54 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

import inspect
import joblib
import json
import math
import numpy as np
import os
import pandas as pd
import typing

from datetime import datetime
from inspect import signature
from io import StringIO
from sklearn import datasets, model_selection, preprocessing, utils
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import mean_squared_error, precision_score, r2_score
from sklearn.metrics import recall_score
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

    Inputs:
    - hyperparameters = set of hyperparameters to test, input with function call
    - model_class = the machine learning model class, input with function call
    - test_set = the data test set, input with function call
    - training_set = the data training set, input with function call
    - validation_set = the data validation set, input with function call

    Internal variables: 
    - filtered_par = a dictionary of hyperparameter names and values
    - grid = parameter grid of hyperparameters
    - model = machine learning model initiated with certain hyperparameters
    - param = variable going through all values of the grid in a for loop
    - performance_metrics_list_values = list of all parameter combinations 
                                        tested
    - performance_metrics_list_RMSE_validation = list of all RMSE validation 
                                                 values
    - RMSE_validation = root mean squared error for the validation set
    - RMSE_validation_0 = comparision value for RMSE_validation
    - y_hat_validation = model prediction

    Returns:
    - best_hyperparameter_values = the values of hyperparameters that produce 
                                   best model
    - best_model = model with the highest RMSE validation value
    - performance_metrics = results of each test for different combinations of 
       hyperparameters
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

    Inputs:
    - model = machine learning model to be tested, input with function call
    - hyperparameters = set of hyperparameters to test, input with function call

    Internal variables: 
    - clf = an object of GridSearchCV initiated with given model and 
            hyperparameters

    Returns: 
    - best_estimator = model with the best result in the GridSearchCV testing
    - best_params = the combination of hyperparameters that produces the best 
                    result
    - best_score = the value of the best result from GridSearchCV
    '''

    clf = GridSearchCV(model, hyperparameters, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)
    best_estimator = clf.best_estimator_
    best_score = clf.best_score_
    best_params = clf.best_params_
    metrics_dict = {"validation_score": best_score}
    # version 3
    y_pred = clf.predict(X_train)
    r2 = r2_score(y_pred, y_train)
    metrics_dict["R^2_score"] = float(r2)
    mse = mean_squared_error(y_pred, y_train)
    rmse = math.sqrt(mse)
    metrics_dict["RMSE"] = float(rmse)
    print("model, best_score, r2, rmse = ", model, best_score, r2, rmse)

    ''' # version 2
    clf_r2 = GridSearchCV(model, hyperparameters, scoring='r2')
    clf_r2.fit(X_train, y_train)
    best_score = clf_r2.best_score_
    metrics_dict["R^2_score"] = float(best_score)
    clf_mse = GridSearchCV(model, hyperparameters, scoring='mean_squared_error')
    clf_mse.fit(X_train, y_train)
    best_score = clf_mse.best_score_
    metrics_dict["RMSE"] = float(best_score)
    '''
    ''' # version 1
    r2 = r2_score(X_train, y_train)
    metrics_dict["R^2_score"] = float(r2)
    mse = mean_squared_error(X_train, y_train)
    rmse = math.sqrt(mse)
    metrics_dict["RMSE"] = float(rmse)
    '''
    return best_estimator, best_params, metrics_dict
# end tune_regression_model_hyperparameters


def save_model(folder, best_model, best_hyperparameter_values, performance_metrics):

    '''
    This function writes model, hyperparameters and performance metrics into 
    files. First it tests whether the function input refers to neural network 
    data or standalone test data.

    Inputs:
    - best_hyperparameter_values = best hyperparameter values, input with 
                                   function call
    - best_model = best model, input with function call
    - folder = the path to the folder where files should be written, input with 
               function call
    - performance_matrics = performance metrics, input with function call

    Internal variables: 
    - best_model_file = name of the file with the best model
    - datetime_now = current time and date
    - file_name = name of the file to be written, with the path
    - folder = folder to write a file into
    - time_string = text format of current time for writing a folder name
    '''

    if (folder[-15:] == "neural_networks"):
        datetime_now = datetime.now()
        time_string = datetime_now.strftime("%Y-%m-%d_%H:%M:%S")
        folder = os.path.join(folder, 'regression', time_string)
        os.mkdir(folder)
        best_model_file = 'model.pt'
    elif(folder[-7:] == 'best_nn'):
        folder = os.path.join('neural_networks', folder)
        best_model_file = 'model.pt'
    else:
        best_model_file = 'model.joblib'
    # end if
    file_name = os.path.join(folder, best_model_file)
    joblib.dump(best_model, file_name)
    file_name = os.path.join(folder, 'hyperparameters.json')
    with open(file_name, 'w') as open_file:
        json.dump(best_hyperparameter_values, open_file)
    file_name = os.path.join(folder, 'metrics.json')
    with open(file_name, 'w') as open_file:
        json.dump(performance_metrics, open_file)
# end save_model


def evaluate_sgdregressor(X, y, X_train, y_train, X_test, y_test):

    '''
    This function evaluates how good the SGDRegressor model is with the Airbnb 
    data.

    Inputs:
    - X, y, X_train, y_train, X_test, y_test = machine learning division of 
      data into training and testing sets & predictions

    Internal variables: 
    - cv_score = cross validation score
    - score_test = testing set score
    - score_train = training set score
    - sgdr = SGDRegressor object
    - y_hat_train, y_hat_test = machine learning predictions
    '''

    # Establish model object and make predictions
    sgdr = SGDRegressor()
    sgdr.fit(X_train, y_train)
    y_hat_train = sgdr.predict(X_train)
    y_hat_test = sgdr.predict(X_test)
    # Compute the key measures of performance for the regression model
    cv_score = cross_val_score(sgdr, X, y, cv = 10)
    #print("CV mean score: ", cv_score.mean())
    #print()
    #print("MSE (train):", mean_squared_error(y_train, y_hat_train))
    #print("MSE (test):", mean_squared_error(y_test, y_hat_test))
    #print()
    #print("RMSE (train):", mean_squared_error(y_train, y_hat_train, squared=False))
    #print("RMSE (test):", mean_squared_error(y_test, y_hat_test, squared=False))
    #print()
    score_train = sgdr.score(X_train, y_train)
    #print("R-squared (train):", score_train)
    score_test = sgdr.score(X_test, y_test)
    #print("R-squared (test):", score_test)
    #print()
# end evaluate_sgdregressor


def evaluate_all_models(task_folder):

    '''
    This function finds the best set of hyperparameters for a series of 
    regression models (Decision Tree Regressor, Random Forest Regressor, 
    and Gradient Boosting Regressor) or classification models (Decision Tree 
    Classifier, Random Forest Classifier, and Gradient Boosting Classifier) 
    running them through either the function 
    tune_regression_model_hyperparameters or 
    tune_classification_model_hyperparameters. These hyperparameters are then 
    used to find the metrics and values of best hyperparameters for each model 
    and save these on the disk. 

    Input variable:
    - task_folder = indicator of whether regression or classification models 
                    are being evaluated; also setting the folder into which 
                    their results are written, input with function call
    Variables: 
    - best_estimator = model with the best hyperparameter values
    - best_params = hyperparameter values that produce the best score
    - best_score = score acquired with the best hyperparameter values
    - folder = path to the location where results are saved into various files
    - hyperparameters = selection of hyperparameter values to be tested and 
                        evaluated
    - model = machine learning model (regressor)
    '''

    def print_and_save(best_estimator, best_params, best_score, task_folder, sub_folder):
        #print("Best estimator = ", best_estimator)
        #print("Best hyperparameter values = ", best_params)
        #print("Best score = ", best_score)
        folder = os.path.join("models", task_folder, sub_folder)
        save_model(folder, best_estimator, best_params, best_score)
    # end print_and_save
    
    model_name_list = [] ; model_list = [] ; hyperparameter_list = []
    if (task_folder == "regression"):
        # Decision Tree Regressor
        model_name_list.append("decision_tree")
        model_list.append(DecisionTreeRegressor())
        hyperparameters = {
            "max_depth": [2, 5, 10, 20],
            "criterion": ["squared_error", "friedman_mse", "poisson"],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 4, 8]
        }
        hyperparameter_list.append(hyperparameters)
        # Random Forest Regressor
        model_name_list.append("random_forest")
        model_list.append(RandomForestRegressor())
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200, 400],
            "criterion": ["squared_error", "friedman_mse", "poisson"],
            "min_samples_leaf": [1, 2, 4], 
            "bootstrap": [True, False] 
        }
        hyperparameter_list.append(hyperparameters)
        # Gradient Boosting Regressor
        model_name_list.append("gradient_boosting")
        model_list.append(GradientBoostingRegressor())
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200, 400],
            "loss": ["squared_error", "absolute_error", "huber"],
            "min_samples_leaf": [1, 2, 4],
            "learning_rate": [0.001, 0.01, 0.1, 0.2]
        }
        hyperparameter_list.append(hyperparameters)
        # SGD Regressor
        model_name_list.append("sgd_regressor")
        model_list.append(SGDRegressor())
        hyperparameters = {
            "alpha": [0.00001, 0.001, 0.1, 10.0],
            "max_iter": [100, 400, 700, 1000], 
            "learning_rate": ['constant'],
            "eta0": [0.01, 0.04, 0.07, 0.1],
            "power_t": [0.1, 0.5, 0.9]
        }
        hyperparameter_list.append(hyperparameters)
    elif (task_folder == "classification"):
        print("Reached classification")
        # Decision Tree Classifier
        model_name_list.append("decision_tree")
        model_list.append(DecisionTreeClassifier())
        hyperparameters = {
            "max_depth": [2, 5, 10, 20],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_leaf": [1, 2, 4], 
            "min_samples_split": [2, 4, 8]
        }
        hyperparameter_list.append(hyperparameters)
        # Random Forest Classifier
        model_name_list.append("random_forest")
        model_list.append(RandomForestClassifier())
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200, 400],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_leaf": [1, 2, 4], 
            "bootstrap": [True, False] 
        }
        hyperparameter_list.append(hyperparameters)
        # Gradient Boosting Classifier
        model_name_list.append("gradient_boosting")
        model_list.append(GradientBoostingClassifier())
        hyperparameters = {
            "n_estimators": [10, 50, 100, 200, 400],
            "criterion": ["friedman_mse", "squared_error"],
            "min_samples_leaf": [1, 2, 4],
            "learning_rate": [0.001, 0.01, 0.1, 0.2]
        }
        hyperparameter_list.append(hyperparameters)
        # Logistic Regression
        model_name_list.append("logistic_regression")
        model_list.append(LogisticRegression())
        hyperparameters = {
            "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
            "max_iter": [10, 50, 100, 200],
            "C": [0.5, 1, 2]
        }
        hyperparameter_list.append(hyperparameters)
        print("Model name list:", model_name_list)
    # end if
    if (task_folder == "regression"):
        for item in range(0, len(model_list)):
            print("regression item = ", item)
            best_estimator, best_params, best_score = tune_regression_model_hyperparameters(model_list[item], hyperparameter_list[item])
            print_and_save(best_estimator, best_params, best_score, task_folder, model_name_list[item])
        # end for
    elif (task_folder == "classification"):
        for item in range(0, len(model_list)):
            print("classification item = ", item)
            best_estimator, best_params, best_score = tune_classification_model_hyperparameters(model_list[item], hyperparameter_list[item])
            print_and_save(best_estimator, best_params, best_score, task_folder, model_name_list[item])
        # end for
    # end for
# end evaluate_all_models


def find_best_sgdregressor(X_train, X_test, X_validation):

    '''
    This function feeds a grid of hyperparameters to regression model tuning 
    function and finds the best set of hyperparameters, i.e. the ones that 
    result in the best score for the model. 

    Inputs:
    - X_train, X_test, X_validation = machine learning division of data

    Internal variables: 
    - best_estimator = model with the best result in the GridSearchCV testing
    - best_hyperparameter_values = the values of hyperparameters that produce 
                                   best model
    - best_params = the combination of hyperparameters that produces the best 
                    result
    - best_score = the value of the best result from GridSearchCV
    - features_labels_tuple = tuple of airbnb data as a Pandas dataframe and 
                              column headers as a list
    - hyperparameters = a selection of ranges of hyperparameters for testing & 
                        finding the best values
    - model = variable to hold a chosen machine learning model
    - performance_metrics = results of each test for different combinations of 
                            hyperparameters
    - test_set = auxiliary variable to hold X_test
    - training_set = auxiliary variable to hold X_train
    - validation_set = auxiliary variable to hold X_validation
    '''

    # Feed the model class and parameters to a function to find the best model
    training_set = X_train
    validation_set = X_validation
    test_set = X_test
    hyperparameters = {
        "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
        "learning_rate": ['constant'],  # constant for eta0; 
                                        # invscaling for power_t
        "eta0": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "power_t": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
    best_model, best_hyperparameter_values, performance_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, training_set, validation_set, test_set, hyperparameters)
    #print("Best model = ", best_model)
    #print("Best hyperparameter values = ", best_hyperparameter_values)
    #print("Performance metrics = ", performance_metrics)
    # Feed the model and parameters to a function to perform SK-learn GridSearchCV
    model = SGDRegressor()
    best_estimator, best_score, best_params = tune_regression_model_hyperparameters(model, hyperparameters)
    #print("Best estimator: " + str(best_estimator))
    #print("Best score: " + str(best_score))
    #print("Best parameters: " + str(best_params))
    folder = os.path.join("models", "regression", "linear_regression")
    save_model(folder, best_estimator, best_hyperparameter_values, performance_metrics)
# end find_best_sgdregressor


def find_best_model(task_folder):

    '''
    This function compares the machine learning models that have been processed 
    by functions tune_regression_model_hyperparameters and 
    tune_classification_model_hyperparameters. 

    Input: 
    - task_folder = indicator of whether regression or classification models 
                    are being evaluated; also setting the folder into which 
                    their results are written, input with function call

    Internal variables:
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

    Returns:
    - hyperparameter_dictionary = hyperparameters of the best model
    - loaded_model = the best model
    - metrics_dictionary = metrics of the best model
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
    chosen_model = score_file[0]
    for one_of_files in score_file:
        with open(one_of_files, 'r') as contents:
            #metrics_value_test = float(json.loads(contents.read()).get('validation_accuracy'))
            #metrics_value_test = float(json.loads(contents.read()).get('validation_score'))
            if (task_folder == "regression"):
                metrics_value_test = float(json.loads(contents.read()).get('RMSE'))
            elif (task_folder == "classification"):
                metrics_value_test = float(json.loads(contents.read()).get('accuracy'))
            # end if
        # end with
        if (metrics_value_test > metrics_value):
            metrics_value = metrics_value_test
            chosen_model = one_of_files
        # end if
    # end for
    score_path = os.path.join(chosen_model[:-13])
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
    This function trains the logistic regression model to predict the 
    "Category" column from the tabular data. Then it calculates various 
    performance values for the training and test sets. 

    Internal variables: 
    - model = initiated logistic regression 
    - y_hat_train, y_hat_test = machine learning predictions
    '''

    # Train logistic regression to predict the category from the tabular data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    # Compute the F1 score, precision score, recall score and accuracy score
    #print("F1 score for train set:  ", f1_score(y_train, y_hat_train, average="macro"))
    #print("F1 score for test set:   ", f1_score(y_test, y_hat_test, average="macro"))
    #print("Precision for train set: ", precision_score(y_train, y_hat_train, average="macro"))
    #print("Precision for test set:  ", precision_score(y_test, y_hat_test, average="macro"))
    #print("Recall for train set:    ", recall_score(y_train, y_hat_train, average="macro"))
    #print("Recall for test set:     ", recall_score(y_test, y_hat_test, average="macro"))
    #print("Accuracy for train set:  ", accuracy_score(y_train, y_hat_train))
    #print("Accuracy for test set:   ", accuracy_score(y_test, y_hat_test))

    # Tune hyperparameters
    hyperparameters = {
        "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
        "max_iter": [10, 50, 100, 200],
        "C": [0.5, 1, 2]
    }
    best_model, best_params, metrics_dict = tune_classification_model_hyperparameters(model, hyperparameters)
    #print("Best model = ", best_model)
    #print("Best hyperparameter values = ", best_params)
    #print("Accuracy = ", metrics_dict)
# end train_and_evaluate_logistic_regression


def tune_classification_model_hyperparameters(model, hyperparameters):

    '''
    This function uses SKLearn's GridSearchCV to perform a grid search.

    Inputs:
    - model = machine learning model to be tested, input with function call
    - hyperparameters = set of hyperparameters to test, input with function call

    Internal variables: 
    - clf = an object of GridSearchCV initiated with given model and 
            hyperparameters

    Returns: 
    - best_estimator = model with the best result in the GridSearchCV testing
    - best_params = the combination of hyperparameters that produces the best 
                    result
    - best_score = the value of the best result from GridSearchCV
    '''

    print("Arrived to tune_classification_model_hyperparameters")
    #clf = GridSearchCV(model, hyperparameters, scoring='neg_mean_squared_error')
    clf = GridSearchCV(model, hyperparameters, scoring='accuracy')
    clf.fit(X_train, y_train)
    best_estimator = clf.best_estimator_
    best_score = clf.best_score_
    best_params = clf.best_params_
    metrics_dict = {"validation_score": best_score}
    y_pred = clf.predict(X_train)
    f1 = f1_score(y_pred, y_train, average='weighted')
    metrics_dict["F1_score"] = float(f1)
    accuracy = accuracy_score(y_train, y_pred)
    metrics_dict["accuracy"] = float(accuracy)
    print("model, best_score, f1, accuracy = ", model, best_score, f1_score, accuracy)
    return best_estimator, best_params, metrics_dict
# end tune_classification_model_hyperparameters


def tune_classification_model_hyperparameters2(model, hyperparameters):

    '''
    This function uses SKLearn's GridSearchCV to perform a grid search to find 
    the best set of hyperparameters. Then it saves the performance values into 
    files. 

    Inputs: 
    - model = machine learning model to be tested, input with function call
    - hyperparameters = set of hyperparameters to test, input with function call

    Internal variable: 
    - clf = an object of GridSearchCV initiated with given model and hyperparameters

    Returns:
    - best_model = model with the best result in the GridSearchCV testing
    - best_params = the combination of hyperparameters that produces the best 
                    result
    - metrics_dict = the accuracy of the best result from GridSearchCV
    '''

    # Grid search for the best model
    clf = GridSearchCV(model, hyperparameters, scoring='accuracy')
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    best_params = clf.best_params_
    metrics_dict = {"validation_score": clf.best_score_}
    #metrics_dict = {"validation_accuracy": clf.best_score_}
    #print('best_model = ', best_model)
    #print('validation_accuracy = ', clf.best_score_)
    # Save the model performance values
    folder = os.path.join("models", "classification", "logistic_regression")
    save_model(folder, best_model, best_params, metrics_dict)
    return best_model, best_params, metrics_dict
# end tune_classification_model_hyperparameters


def split_data(df):

    '''
    This function divides data in a Pandas dataframe into training, testing 
    and validation sets.

    Input:
    - df = Pandas dataframe containing the data

    Returns:
    - X, y, X_train, y_train, X_test, y_test, X_validation, y_validation
      = machine learning division of input data into training, testing and validation sets
    '''

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    return X, y, X_train, y_train, X_test, y_test, X_validation, y_validation
# end split_data



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
features_labels_tuple = load_airbnb("bedrooms")
# Using sklearn to train a linear regression model to predict first 
# the "Price_Night" and then the "Category" feature from the tabular data.
df_in = features_labels_tuple[0]
df = df_in.replace(['Treehouses', 'Category', 'Chalets', 'Amazing pools', 'Offbeat', 'Beachfront'],
                   [1, 1, 2, 3, 4, 5])
X, y, X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(df)
if __name__ == "__main__":
    # Evaluate SGDregressor model
    evaluate_sgdregressor(X, y, X_train, y_train, X_test, y_test)
    find_best_sgdregressor(X_train, X_test, X_validation)
    # Evaluate a set of alternative regression models
    evaluate_all_models("regression")
    # Find the best regression model from those processed earlier
    loaded_model, hyperparameter_dictionary, metrics_dictionary = find_best_model("regression")
    #print("loaded model = ", loaded_model)
    #print("hyperparameter dictionary = ", hyperparameter_dictionary)
    #print("metrics dictionary = ", metrics_dictionary)
    # Train and evaluate logistic regression
    train_and_evaluate_logistic_regression()
    # Evaluate a set of alternative classification models
    evaluate_all_models("classification")
    # Find the best classification model from those processed earlier
    loaded_model, hyperparameter_dictionary, metrics_dictionary = find_best_model("classification")
    #print("loaded model = ", loaded_model)
    #print("hyperparameter dictionary = ", hyperparameter_dictionary)
    #print("metrics dictionary = ", metrics_dictionary)
# end if
