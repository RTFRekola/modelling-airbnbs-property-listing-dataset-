#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Tue 21 Nov 2023 at 11:55 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from tabular_data import load_airbnb

import yaml


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
    This class 

    Variables: 

    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define layers.
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, config['hidden_layer_width']),
            torch.nn.ReLU(),
            torch.nn.Linear(config['hidden_layer_width'], config['hidden_layer_width']),
            torch.nn.ReLU(),
            torch.nn.Linear(config['hidden_layer_width'], config['hidden_layer_width']),
            torch.nn.ReLU(),
            torch.nn.Linear(config['hidden_layer_width'], 1)
        )
        #self.layers = torch.nn.Sequential(
        #    torch.nn.Linear(11, config['hidden_layer_width']),
        #    for step in range(config(model_depth)-1):
        #        torch.nn.ReLU(),
        #        torch.nn.Linear(config['hidden_layer_width'], config['hidden_layer_width']),
        #    # end for
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(config['hidden_layer_width'], 1)
        #)

    def forward(self, X):
        # Return prediction.
        return self.layers(X)

# end Neural_Network    


def train(model, loader, epochs, config):

    '''
    This function does training on the neural network model. 

    Variables: 
    - epochs = the number of epochs to be processed, input with function call
    - model = the instantiated neural network model to be used, input with function call
    - loader = the dataloader, input with function call

    - batch = loop parameter
    - epoch = loop parameter
    - features = numerical Airbnb features
    - labels = price per night
    - prediction = the neural network output
    '''

    #optimiser = torch.optim.SGD(model.parameters(), lr=0.0002)
    #optimiser = config['optimiser'](model.parameters(), lr=config['learning_rate'])
    optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in loader:
            features, labels = batch
            labels = labels.unsqueeze(1)
            prediction = model(features)
            loss = f.mse_loss(prediction.float(), labels.float())
            loss.backward()
            print(loss.item())
            # Optimisation step
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
        # end for
    # end for
# end train


def get_nn_config():
    with open("nn_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
# end get_nn_config


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
    hyperparameters = get_nn_config()

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
    train(model, train_loader, epochs, hyperparameters)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=128)
    train(model, validation_loader, epochs, hyperparameters)
    '''
    # Instantiate neural network model and train it for training set.
    model = Neural_Network(hyperparameters)
    epochs = 10
    train(model, train_loader, epochs, hyperparameters)

# end if

### end programme
