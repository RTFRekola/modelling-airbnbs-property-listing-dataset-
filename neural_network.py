#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Tue 05 Dec 2023 at 19:42 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

from datetime import datetime
import numpy as np
import os
import pandas as pd
import time
import yaml

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import scale

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import R2Score

from modelling import save_model
from tabular_data import load_airbnb


# ==============================================
# ===   Functions   ============================
# ==============================================


class AirbnbNightlyPriceRegressionDataset(Dataset):

    '''
    This class takes the Airbnb data and returns numerical Airbnb features and price per night labels. 

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


class LinearRegression(torch.nn.Module):

    '''
    This class takes the numerical Airbnb features and returns a Linear Regression model on these 
    features.

    Variables: 
    self.layers = sequence of applications of the model to cover the depth of the neural network
    self.linear_layer = the Torch Linear Regression model (for the exact number of features we have)
    '''

    def __init__(self):
        super().__init__()
        # Initialise parameters.
        self.linear_layer = torch.nn.Linear(11, 1)

    def __call__(self, features):
        # Use the layers of transformation to proceess the features.
        return self.linear_layer(features)

# end LinearRegression


class Neural_Network(torch.nn.Module):

    '''
    This class tests neural network models with different parameters settings. 

    Variables: 
    i = loop parameter, input with function call
    self.config = hyperparameter dictionary, input with function call

    layers = list of instantiated models for Sequential, internal in get_net
    step = loop parameter, internal in get_net

    self.layers = sequence of applications of the model to cover the depth of the neural network
    '''

    def __init__(self, config, i_width, i_depth):
        super().__init__()
        self.config = config
        self.i_width = i_width
        self.i_depth = i_depth
        

        # Define layers.
        self.layers = torch.nn.Sequential(*self.get_net(self.i_width, self.i_depth))

    def get_net(self, i_width, i_depth):
        layers = [torch.nn.Linear(11, self.config['hidden_layer_width'][i_width]), torch.nn.ReLU()]
        for step in range(self.config['model_depth'][i_depth]-1):
            layers.append(torch.nn.Linear(self.config['hidden_layer_width'][i_width], 
                                          self.config['hidden_layer_width'][i_width]))
            layers.append(torch.nn.ReLU())
        # end for
        layers.append(torch.nn.Linear(self.config['hidden_layer_width'][i_width], 1))
        return layers

    def forward(self, X):
        # Return prediction.
        return self.layers(X)

# end Neural_Network    


def train(model, loader, epochs, config, i, j):

    '''
    This function does training on the neural network model. 

    Variables: 
    - epochs = the number of epochs to be processed, input with function call
    - i = loop counter for going through optimisers, input with function call
    - j = loop counter for going through values of learning rate, input with function call
    - model = the instantiated neural network model to be used, input with function call
    - loader = the dataloader, input with function call

    - batch = loop parameter
    - epoch = loop parameter
    - features = numerical Airbnb features
    - labels = price per night
    - prediction = the neural network output
    '''

    if (config['optimiser'][i] == 'SGD'):
        optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'][j])
    elif (config['optimiser'][i] == 'Adagrad'):
        optimiser = torch.optim.Adagrad(model.parameters(), lr=config['learning_rate'][j])
    elif (config['optimiser'][i] == 'Adam'):
        optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'][j])
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


def find_best_nn(config):

    '''
    This function goes through all different combinations of hyperparameters and tests them 
    in the function 'train'.

    Variables: 
    - config = dictionary with all the hyperparameters in it, input with function call

    - datetime_beg = time at the beginning of training
    - datetime_end = time at the end of training
    - epochs = number of epochs
    - folder = name of the folder results are being saved in
    - hyperparameters = dictionary with current set of hyperparameters
    - i_depth = loop counter to go through all the depth values in hyperparameters
    - i_opt = loop counter to go through all the optimisers in hyperparameters
    - i_rate = loop counter to go through all the learning rates in hyperparameters
    - i_width = loop counter to go through all the hidden layer widths in hyperparameters
    - inference_latency = average time taken to make a prediction
    - loss_function_baseline = comparison point with always the lowest loss in it
    - metrics = dictionary with performance metrics in it
    - model = model fetched from Neural_Network
    - r2_score = R^2 score
    - rmse_loss = RMSE loss
    - training_duration = time taken to train the model
    '''

    epochs = 10
    loss_function_baseline = 999.9
    #print('len(optimiser) = ', len(config['optimiser']))
    for i_opt in range(len(config['optimiser'])):
        #print('i_opt = ', i_opt)
        #print('len(hidden_layer_width = ', len(config['hidden_layer_width']))
        for i_width in range(len(config['hidden_layer_width'])):
            #print('i_width = ', i_width)
            #print('len(model_depth) = ', len(config['model_depth']))
            for i_depth in range(len(config['model_depth'])):
                #print('i_depth = ', i_depth)
                #print('len(learning_rate) = ', len(config['learning_rate']))
                for i_rate in range(len(config['learning_rate'])):
                    #print('i_rate = ', i_rate)
                    datetime_beg = datetime.now()

                    hyperparameters = {'optimiser': config['optimiser'][i_opt]}
                    hyperparameters['hidden_layer_width'] = config['hidden_layer_width'][i_width]
                    hyperparameters['model_depth'] = config['model_depth'][i_depth]
                    model = Neural_Network(config, i_width, i_depth)

                    hyperparameters['learning_rate'] = config['learning_rate'][i_rate]
                    rmse_loss, r2_score, inference_latency = train(model, train_loader, 
                                                                   epochs, config, i_opt, i_rate)
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

                    # Add one second delay, just in case someone runs this with a very fast computer 
                    # and the result folders (with time stamps) would end up having the same second 
                    # or a name with that same second.
                    time.sleep(1)  # Sleep for one second.
                # end for
            # end for
        # end for
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

# Load the input data.
features_labels_tuple = load_airbnb()

# Using sklearn to train a linear regression model to predict first 
# the "Price_Night" and then the "Category" feature from the tabular data.
df_in = features_labels_tuple[0]
df = df_in.replace(['Treehouses', 'Category', 'Chalets', 'Amazing pools', 'Offbeat', 'Beachfront'],
                   [1, 1, 2, 3, 4, 5])
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)

if __name__ == '__main__':
    # Load hyperparameters from file.
    config = get_nn_config()

    # Prepare the neural network dataset.
    dataset = AirbnbNightlyPriceRegressionDataset()

    train_len = int(len(dataset)*0.8)      
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len])

    # Prepare the neural network dataloaders.
    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=128)

    train_len = int(len(train_set)*0.75)      
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_len, 
                                                                          len(train_set)-train_len])

    # Prepare the features and labels for the neural network model.
    example = next(iter(train_loader))
    features, labels = example
    '''
    # Instantiate the model and train it both for training set and validation set
    model = LinearRegression()
    epochs = 10
    train(model, train_loader, epochs, config)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=128)
    train(model, validation_loader, epochs, config)
    '''
    # Instantiate neural network model and train it for a training set with a selection of 
    # hyperparameters. Save results of each run.

    '''
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
    '''

    find_best_nn(config)
# end if

### end programme
