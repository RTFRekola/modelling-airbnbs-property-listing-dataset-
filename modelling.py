#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Sat 29 Jun 2024 at 12:58 UT 

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
    y_pred = clf.predict(X_train)
    r2 = r2_score(y_pred, y_train)
    metrics_dict["R^2_score_train"] = float(r2)
    mse = mean_squared_error(y_pred, y_train)
    rmse = math.sqrt(mse)
    metrics_dict["RMSE_train"] = float(rmse)
    y_pred_test = clf.predict(X_test)
    r2_test = r2_score(y_pred_test, y_test)
    metrics_dict["R^2_score_test"] = float(r2_test)
    mse_test = mean_squared_error(y_pred_test, y_test)
    rmse_test = math.sqrt(mse_test)
    metrics_dict["RMSE_test"] = float(rmse_test)
    y_pred_validation = clf.predict(X_validation)
    r2_validation = r2_score(y_pred_validation, y_validation)
    metrics_dict["R^2_score_validation"] = float(r2_validation)
    mse_validation = mean_squared_error(y_pred_validation, y_validation)
    rmse_validation = math.sqrt(mse_validation)
    metrics_dict["RMSE_validation"] = float(rmse_validation)
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
    score_train = sgdr.score(X_train, y_train)
    score_test = sgdr.score(X_test, y_test)
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
        folder = os.path.join("models", task_folder, sub_folder)
        save_model(folder, best_estimator, best_params, best_score)
    # end print_and_save
    
    model_name_list_r = [] ; model_list_r = [] ; hyperparameter_list_r = []
    model_name_list_c = [] ; model_list_c = [] ; hyperparameter_list_c = []
    if (task_folder == "regression"):
        # Decision Tree Regressor
        model_name_list_r.append("decision_tree")
        model_list_r.append(DecisionTreeRegressor())
        hyperparameters = {
            "ccp_alpha": [0.3, 0.9, 3.0], 
            "criterion": ["squared_error", "friedman_mse", "poisson"],
            "max_depth": [1, 2],
            "max_features": [1, 2], 
            "min_impurity_decrease": [0.5, 0.9, 2.0, 5.0], 
            "min_samples_leaf": [10, 15],
            "min_samples_split": [1, 2, 3], 
            "min_weight_fraction_leaf": [0.3, 0.5]
        }
        hyperparameter_list_r.append(hyperparameters)
        # Random Forest Regressor
        model_name_list_r.append("random_forest")
        model_list_r.append(RandomForestRegressor())
        hyperparameters = {
            "bootstrap": [True, False], 
            "ccp_alpha": [0.99, 2, 3], 
            "criterion": ["squared_error", "friedman_mse", "poisson"],
            "max_depth": [1, 2],
            "max_features": [1, 2], 
            "min_samples_leaf": [1, 2, 4], 
            "min_samples_split": [15, 20], 
            "n_estimators": [10, 100]
        }
        hyperparameter_list_r.append(hyperparameters)
        # Gradient Boosting Regressor
        model_name_list_r.append("gradient_boosting")
        model_list_r.append(GradientBoostingRegressor())
        hyperparameters = {
            "learning_rate": [0.01, 0.05], 
            "loss": ["squared_error", "absolute_error", "huber"],
            "max_depth": [1, 2],
            "max_features": [1, 2], 
            "min_impurity_decrease": [0.2, 0.3], 
            "min_samples_leaf": [6, 10],
            "min_samples_split": [15, 20], 
            "n_estimators": [5, 10]
        }
        hyperparameter_list_r.append(hyperparameters)
        # SGD Regressor
        model_name_list_r.append("sgd_regressor")
        model_list_r.append(SGDRegressor())
        hyperparameters = {
            "alpha": [0.00001, 0.0001, 0.001],
            "eta0": [0.00001, 0.0001, 0.001],
            "learning_rate": ['constant'],
            "max_iter": [1000, 5000, 10000], 
            "penalty": ['l2', 'l1', 'elasticnet', 'None'], 
            "power_t": [0.3, 0.5, 0.9]
        }
        hyperparameter_list_r.append(hyperparameters)
    elif (task_folder == "classification"):
        print("Reached classification")
        # Decision Tree Classifier
        model_name_list_c.append("decision_tree")
        model_list_c.append(DecisionTreeClassifier())
        hyperparameters = {
            "max_depth": [1, 2, 3, 6, 10, 15, 25],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_leaf": [1, 2, 4, 8, 12, 16, 32, 64], 
            "min_samples_split": [1, 2, 4, 8, 12, 16, 32, 64]
        }
        hyperparameter_list_c.append(hyperparameters)
        # Random Forest Classifier
        model_name_list_c.append("random_forest")
        model_list_c.append(RandomForestClassifier())
        hyperparameters = {
            "n_estimators": [5, 10, 20, 50, 100, 200, 500, 1000, 2000],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_leaf": [1, 2, 4, 8, 12, 16, 32], 
            "bootstrap": [True] 
        }
        hyperparameter_list_c.append(hyperparameters)
        # Gradient Boosting Classifier
        model_name_list_c.append("gradient_boosting")
        model_list_c.append(GradientBoostingClassifier())
        hyperparameters = {
            "n_estimators": [100, 200, 500, 1000, 2000, 4000, 8000],
            "criterion": ["friedman_mse", "squared_error"],
            "min_samples_leaf": [1, 2, 4, 8, 12, 16],
            "learning_rate": [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        }
        hyperparameter_list_c.append(hyperparameters)
        # Logistic Regression
        model_name_list_c.append("logistic_regression")
        model_list_c.append(LogisticRegression())
        hyperparameters = {
            "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
            "max_iter": [2, 5, 10, 50, 100],
            "C": [0.01, 0.1, 0.5, 1, 2, 4, 8]
        }
        hyperparameter_list_c.append(hyperparameters)
        print("Model name list:", model_name_list_c)
    # end if
    if (task_folder == "regression"):
        for item in range(0, len(model_list_r)):
            print("regression item = ", item)
            best_estimator, best_params, best_score = tune_regression_model_hyperparameters(model_list_r[item], hyperparameter_list_r[item])
            print_and_save(best_estimator, best_params, best_score, task_folder, model_name_list_r[item])
        # end for
    elif (task_folder == "classification"):
        for item in range(0, len(model_list_c)):
            print("classification item = ", item)
            best_estimator, best_params, best_score = tune_classification_model_hyperparameters(model_list_c[item], hyperparameter_list_c[item])
            print_and_save(best_estimator, best_params, best_score, task_folder, model_name_list_c[item])
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
        "max_iter": [500, 1000, 5000, 10000, 50000], 
        "learning_rate": ['constant'],  # constant for eta0; 
                                        # invscaling for power_t
        "eta0": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "power_t": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }
    best_model, best_hyperparameter_values, performance_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, training_set, validation_set, test_set, hyperparameters)
    model = SGDRegressor()
    best_estimator, best_score, best_params = tune_regression_model_hyperparameters(model, hyperparameters)
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
    y_hat_validation = model.predict(X_validation)
    # Compute the F1 score, precision score, recall score and accuracy score
    print("F1 score for train set:  ", f1_score(y_train, y_hat_train, average="macro"))
    print("F1 score for test set:   ", f1_score(y_test, y_hat_test, average="macro"))
    print("F1 score for validation set: ", f1_score(y_validation, y_hat_validation, average="macro"))
    print("Precision for train set: ", precision_score(y_train, y_hat_train, average="macro"))
    print("Precision for test set:  ", precision_score(y_test, y_hat_test, average="macro"))
    print("Precision for validation set: ", precision_score(y_validation, y_hat_validation, average="macro"))
    print("Recall for train set:    ", recall_score(y_train, y_hat_train, average="macro"))
    print("Recall for test set:     ", recall_score(y_test, y_hat_test, average="macro"))
    print("Recall for validation set: ", recall_score(y_validation, y_hat_validation, average="macro"))
    print("Accuracy for train set:  ", accuracy_score(y_train, y_hat_train))
    print("Accuracy for test set:   ", accuracy_score(y_test, y_hat_test))
    print("Accuracy for validation set: ", accuracy_score(y_validation, y_hat_validation))
    # Tune hyperparameters
    hyperparameters = {
        "solver": ["newton-cg", "newton-cholesky", "sag", "saga"],
        "max_iter": [10, 50, 100, 200],
        "C": [0.5, 1, 2]
    }
    best_model, best_params, metrics_dict = tune_classification_model_hyperparameters(model, hyperparameters)
    print("Best model = ", best_model)
    print("Best hyperparameter values = ", best_params)
    print("Accuracy = ", metrics_dict)
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

    clf = GridSearchCV(model, hyperparameters, scoring='accuracy', cv=5)
    clf.fit(X_train, y_train)
    best_estimator = clf.best_estimator_
    best_score = clf.best_score_
    best_params = clf.best_params_
    metrics_dict = {"validation_score": best_score}
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_validation = clf.predict(X_validation)
    f1_train = f1_score(y_pred_train, y_train, average='weighted')
    f1_test = f1_score(y_pred_test, y_test, average='weighted')
    f1_validation = f1_score(y_pred_validation, y_validation, average='weighted')
    metrics_dict["F1_score_train"] = float(f1_train)
    metrics_dict["F1_score_test"] = float(f1_test)
    metrics_dict["F1_score_validation"] = float(f1_validation)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_validation = accuracy_score(y_validation, y_pred_validation)
    metrics_dict["accuracy_train"] = float(accuracy_train)
    metrics_dict["accuracy_test"] = float(accuracy_test)
    metrics_dict["accuracy_validation"] = float(accuracy_validation)
    recall_train = recall_score(y_train, y_pred_train, average="macro")
    recall_test = recall_score(y_test, y_pred_test, average="macro")
    recall_validation = recall_score(y_validation, y_pred_validation, average="macro")
    metrics_dict["recall_score_train"] = float(recall_train)
    metrics_dict["recall_score_test"] = float(recall_test)
    metrics_dict["recall_score_validation"] = float(recall_validation)
    precision_train = precision_score(y_train, y_pred_train, average="macro")
    precision_test = precision_score(y_test, y_pred_test, average="macro")
    precision_validation = precision_score(y_validation, y_pred_validation, average="macro")
    metrics_dict["precision_score_train"] = float(precision_train)
    metrics_dict["precision_score_test"] = float(precision_test)
    metrics_dict["precision_score_validation"] = float(precision_validation)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
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

if __name__ == "__main__":
    features_labels_tuple = load_airbnb("Price_Night")
    #features_labels_tuple = load_airbnb("Category")
    #features_labels_tuple = load_airbnb("bedrooms")
    df = features_labels_tuple[0]
    X, y, X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(df)
    # Evaluate SGDregressor model
    #evaluate_sgdregressor(X, y, X_train, y_train, X_test, y_test)
    #find_best_sgdregressor(X_train, X_test, X_validation)
    # Evaluate a set of alternative regression models
    evaluate_all_models("regression")
    # Find the best regression model from those processed earlier
    #loaded_model, hyperparameter_dictionary, metrics_dictionary = find_best_model("regression")
    #print("loaded model = ", loaded_model)
    #print("hyperparameter dictionary = ", hyperparameter_dictionary)
    #print("metrics dictionary = ", metrics_dictionary)
    # Train and evaluate logistic regression
    #train_and_evaluate_logistic_regression()
    # Evaluate a set of alternative classification models
    #evaluate_all_models("classification")
    # Find the best classification model from those processed earlier
    #loaded_model, hyperparameter_dictionary, metrics_dictionary = find_best_model("classification")
    #print("loaded model = ", loaded_model)
    #print("hyperparameter dictionary = ", hyperparameter_dictionary)
    #print("metrics dictionary = ", metrics_dictionary)
# end if
