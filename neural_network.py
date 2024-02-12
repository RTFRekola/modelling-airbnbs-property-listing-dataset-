#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Thu 8 Feb 2024 at 15:29 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

import itertools
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as f
import yaml

from datetime import datetime
from modelling import save_model, split_data
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import scale
from tabular_data import load_airbnb
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import R2Score


# ==============================================
# ===   Functions   ============================
# ==============================================


class AirbnbNightlyPriceRegressionDataset(Dataset):

    '''
    This class takes the Airbnb data and returns numerical Airbnb features and 
    price per night labels. 

    Variables: 
    example = Airbnb data on the row "index"
    self.features = the numerical Airbnb features of the house
    self.label = the price per night of the house
    '''

    def __init__(self):
        super().__init__()
        self.data = features_labels_tuple[0]

    def __getitem__(self, index):
        example = self.data.iloc[index]
        self.features = example[1:]
        self.label = example[0]
        return (torch.tensor(self.features).float(), self.label)

    def __len__(self):
        return len(self.data)
# end AirbnbNightlyPriceRegressionDataset


class Neural_Network(torch.nn.Module):

    '''
    This class tests neural network models with different parameters settings. 

    Variables: 
    layers = list of instantiated models for Sequential, internal in get_net
    self.config = hyperparameter dictionary, input with function call
    self.depth = model depth
    self.layers = sequence of applications of the model to cover the depth of 
                  the neural network
    self.width = hidden layer width
    step = loop parameter, internal in get_net
    '''

    def __init__(self, width_value, depth_value):
        super().__init__()
        self.width_value = width_value
        self.depth_value = depth_value
        
        # Define layers.
        self.layers = torch.nn.Sequential(*self.get_net(self.width_value, self.depth_value))

    def get_net(self, width_value, depth_value):
        layers = [torch.nn.Linear(12, width_value), torch.nn.ReLU()]
        for step in range(depth_value-1):
            layers.append(torch.nn.Linear(width_value, width_value))
            layers.append(torch.nn.ReLU())
        # end for
        layers.append(torch.nn.Linear(width_value, 1))
        return layers

    def forward(self, X):
        # Return prediction.
        return self.layers(X)
# end Neural_Network    


def train(model, loader, epochs, optimiser_value, learning_rate_value):

    '''
    This function does training on the neural network model. 

    Inputs:
    - epochs = the number of epochs to be processed, input with function call
    - model = the instantiated neural network model to be used, input with 
              function call
    - optimiser_value = optimiser
    - learning_rate_value = learning rate
    - loader = the dataloader, input with function call

    Internal variables: 
    - batch = loop parameter
    - epoch = loop parameter
    - features = numerical Airbnb features
    - labels = price per night
    - prediction = the neural network output

    Returns: 
    - inference_latency = average time taken to make a prediction
    - r2_score = the R squared score
    - rmse_loss_average = the average RMSE loss
    '''

    if (optimiser_value == 'SGD'):
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate_value)
    elif (optimiser_value == 'Adagrad'):
        optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate_value)
    elif (optimiser_value == 'Adam'):
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate_value)
    # end if
    writer = SummaryWriter()
    batch_idx = 0
    inference_latency_items = []
    rmse_loss_items = []
    r2_score_items = []
    for epoch in range(epochs):
        for batch in loader:
            features, labels = batch
            labels = labels.unsqueeze(1)
            # Find the time it takes to make the prediction
            datetime_before = datetime.now()
            prediction = model(features)
            datetime_after = datetime.now()
            inference_latency_items.append((datetime_after - datetime_before).total_seconds())
            # Find the RMSE loss
            loss_function = torch.nn.MSELoss()
            rmse_loss = torch.sqrt(loss_function(prediction.float(), labels.float()))
            rmse_loss.backward()
            rmse_loss_items.append(rmse_loss)
            # Find the R^2 score
            r2_metric = R2Score()
            r2_metric.update(prediction.float(), labels.float())
            r2_score_items.append(r2_metric.compute())
            # Optimisation step
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', rmse_loss.item(), batch_idx)
            batch_idx += 1
        # end for
    # end for
    # Finalise the values for metrics 
    r2_score = sum(r2_score_items) / len(r2_score_items)
    rmse_loss_average = sum(rmse_loss_items) / len(rmse_loss_items)
    inference_latency = sum(inference_latency_items) / len(inference_latency_items)
    return rmse_loss_average, r2_score, inference_latency
# end train


def get_nn_config():
    '''
    This function reads in a yaml file with hyperparameter values and returns it as a dictionary.
    '''

    with open("nn_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
# end get_nn_config


def find_best_nn(config, train_loader):

    '''
    This function goes through all different combinations of hyperparameters 
    and tests them in the function 'train'.

    Input: 
    - config = dictionary with all the hyperparameters in it, input with 
               function call
    - train_loader = dataloader for the training set

    Internal variables: 
    - datetime_beg = time at the beginning of training
    - datetime_end = time at the end of training
    - epochs = number of epochs
    - folder = name of the folder results are being saved in
    - hidden_layer_width_list = list containing all hidden layer widths to test
    - hyperparameters = dictionary with current set of hyperparameters
    - inference_latency = average time taken to make a prediction
    - itertools_list = list containing all hyperparameter lists
    - itertools_permutations = list containing all possible combinations of 
                               hyperparameters
    - learning_rate_list = list containing all learning rates to test
    - loss_function_baseline = comparison point with always the lowest loss 
                               stored in it
    - metrics = dictionary with performance metrics in it
    - model = model fetched from Neural_Network
    - model_depth_list = list containing all model depths to test
    - optimiser_list = list containing all optimisers to test
    - r2_score = R^2 score
    - rmse_loss = RMSE loss
    - training_duration = time taken to train the model
    '''

    epochs = 10
    loss_function_baseline = 999.9
    optimiser_list = config['optimiser']
    hidden_layer_width_list = config['hidden_layer_width']
    model_depth_list = config['model_depth']
    learning_rate_list = config['learning_rate']
    itertools_list = [optimiser_list, hidden_layer_width_list, 
                      model_depth_list, learning_rate_list]
    itertools_permutations = list(itertools.product(*itertools_list))
    for item in itertools_permutations:
        datetime_beg = datetime.now()
        hyperparameters = {'optimiser': item[0]}
        hyperparameters['hidden_layer_width'] = item[1]
        hyperparameters['model_depth'] = item[2]
        hyperparameters['learning_rate'] = item[3]
        model = Neural_Network(item[1], item[2])
        rmse_loss, r2_score, inference_latency = train(model, train_loader, 
                                                       epochs, item[0], 
                                                       item[3])
        datetime_end = datetime.now()
        training_duration = (datetime_end - datetime_beg).total_seconds()
        metrics = {'RMSE_loss': float(rmse_loss)}
        metrics['R^2 score'] = float(r2_score)
        metrics['training_duration'] = float(training_duration)
        metrics['inference_latency'] = float(inference_latency)
        folder = 'neural_networks'
        save_model(folder, model, hyperparameters, metrics)
        if (rmse_loss < loss_function_baseline):
            folder = 'best_nn'
            save_model(folder, model, hyperparameters, metrics)
            loss_function_baseline = rmse_loss
        # end if

        # Add one second delay, just in case someone runs this with 
        # a very fast computer and the result folders (with time 
        # stamps) would end up having the same second 
        # or a name with that same second.
        time.sleep(1)  # Sleep for one second.
    # end for
# end find_best_nn


# ==============================================
# ===   Main programme   =======================
# ==============================================

'''
Go through the steps to produce final results and call functions as needed.

Parameters:
- dataset = neural network dataset
- df_in = Pandas dataframe with Airbnb data in it
- df = Pandas dataframe, where "Category" values have been converted into 
       numerical values
- model = chosen model for neural network
- run_type = a flag to choose between two options
- test_loader = dataloader for the test set
- test_set = test set for neural network
- train_len = desired fraction of the length of dataset for training set 
              divisions
- train_loader = dataloader for the training set
- train_set = training set for neural network
- validation_set = validation set for the neural network
- X, y, X_train, y_train, X_test, y_test, X_validation, y_validation
   = machine learning division of input data into training, testing and 
     validation sets
'''

# Load the input data.
#features_labels_tuple = load_airbnb("Price_Night")
features_labels_tuple = load_airbnb("bedrooms")
#features_labels_tuple = load_airbnb("Category")
# Using sklearn to train a linear regression model to predict a feature from 
# the tabular data. Keep one of the lines above uncommented and comment the 
# other two. 
df = features_labels_tuple[0]
X, y, X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(df)
print(df.dtypes)
if __name__ == '__main__':
    # Load hyperparameters from file.
    config = get_nn_config()
    # Prepare the neural network dataset.
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_len = int(len(dataset)*0.8)      
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len])
    # Prepare the neural network dataloaders.
    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
    train_len = int(len(train_set)*0.75)      
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_len, len(train_set)-train_len])
    # Prepare the features and labels for the neural network model.
    example = next(iter(train_loader))
    features, labels = example
    # Run tests on the data, uncomment one of the lines below, comment the 
    # other one
    #run_type = "individual"
    run_type = "neural_network"
    if (run_type == "individual"):
        # Use this for individual test runs; the following "find_best_nn" is 
        # for complex runs
        epochs = 10
        model = Neural_Network(config, 1, 1)
        grid = GridSearchCV(estimator=model, param_grid=config)
        grid_result = grid.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    elif (run_type == "neural_network"):
        # Instantiate neural network model and train it for a training set with 
        # a selection of hyperparameters. Save results of each run.
        find_best_nn(config, train_loader)
    # end if
# end if

### end programme
