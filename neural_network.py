#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Sun 15 Aug 2024 at 12:07 UT 

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
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
        self.features = torch.tensor(example[1:]).float()
        self.label = torch.tensor(example[0]).long()
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
    self.depth_value = model depth
    self.layers = sequence of applications of the model to cover the depth of 
                  the neural network
    self.width_value = hidden layer width
    step = loop parameter, internal in get_net
    '''

    def __init__(self, width_value, depth_value, number_of_classes):
        super().__init__()
        self.width_value = width_value
        self.depth_value = depth_value
        self.number_of_classes = number_of_classes
        
        # Define layers.
        self.layers = torch.nn.Sequential(*self.get_net(self.width_value, 
self.depth_value, self.number_of_classes))

    def get_net(self, width_value, depth_value, number_of_classes):
        layers = [torch.nn.Linear(12, width_value), torch.nn.ReLU()]
        for step in range(depth_value-1):
            layers.append(torch.nn.Linear(width_value, width_value))
            layers.append(torch.nn.ReLU())
        # end for
        layers.append(torch.nn.Linear(width_value, number_of_classes))
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
            inference_latency_items.append((datetime_after - 
                                            datetime_before).total_seconds())
            # Find the RMSE loss
            loss_function = torch.nn.MSELoss()
            rmse_loss = torch.sqrt(loss_function(prediction.float(), 
                                                 labels.float()))
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
#    print('rmse_max = ', max(rmse_loss_items))
    return rmse_loss_average, r2_score, inference_latency
# end train


def train_classifier(model, loader, epochs, optimiser_value, learning_rate_value):

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
    - accuracy = average accuracy score
    - f1_score = average F1 score
    - precision = average precision score
    - recall = average recall score
    '''

    if (optimiser_value == 'SGD'):
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate_value)
    elif (optimiser_value == 'Adagrad'):
        optimiser = torch.optim.Adagrad(model.parameters(), 
                                        lr=learning_rate_value)
    elif (optimiser_value == 'Adam'):
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate_value)
    # end if
    writer = SummaryWriter()
    batch_idx = 0
    inference_latency_items = []
    cross_entropy_items = []
    accuracy_items = []
    f1_score_items = []
    precision_items = []
    recall_items = []
    for epoch in range(epochs):
        i = 0
        for batch in loader:
            i += 1
            features, labels = batch
            labels = labels.squeeze()
            # Find the time it takes to make the prediction
            datetime_before = datetime.now()
            prediction = model(features)
            datetime_after = datetime.now()
            labels0 = labels ; prediction_tensor0 = prediction
            inference_latency_items.append((datetime_after - 
                                            datetime_before).total_seconds())
            # Find cross entropy
            loss = f.cross_entropy(prediction, labels)
            cross_entropy_items.append(loss.item())
            _, predicted = torch.max(prediction, 1)
            prediction = prediction.detach().numpy()
            labels = labels.numpy()
            prediction = predicted.numpy()
            # Find accuracy
            accuracy_items.append(accuracy_score(labels, prediction))
            # Find the F1 score
            f1_score_items.append(f1_score(labels, prediction, average='macro'))
            # Find precision
            precision_items.append(precision_score(labels, prediction, 
                                                   average='macro'))
            # Find recall
            recall_items.append(recall_score(labels, prediction, 
                                             average='macro'))
            # Optimisation step
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
        # end for
    # end for
    # Finalise the values for metrics 
    cross_entropy_value = sum(cross_entropy_items) / len(cross_entropy_items)
    accuracy = sum(accuracy_items) / len(accuracy_items)
    f1_score_value = sum(f1_score_items) / len(f1_score_items)
    precision = sum(precision_items) / len(precision_items)
    recall = sum(recall_items) / len(recall_items)
    inference_latency = (sum(inference_latency_items) / 
                         len(inference_latency_items))
    return cross_entropy_value, accuracy, f1_score_value, precision, recall, inference_latency
# end train_classifier


def get_nn_config():
    '''
    This function reads in a yaml file with hyperparameter values and returns it as a dictionary.
    '''

    with open("nn_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
# end get_nn_config


def find_best_nn(config, train_loader, the_label, number_of_classes):

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

    epochs = 200
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
        if (the_label == "Category"):
            model = Neural_Network(item[1], item[2], number_of_classes)
            cross_entropy_value, accuracy, f1_score_value, precision, recall, inference_latency = train_classifier(model, train_loader, epochs, item[0], item[3])
        else:
            model = Neural_Network(item[1], item[2], 1)
            rmse_loss, r2_score, inference_latency = train(model, train_loader, 
                                                       epochs, item[0], 
                                                       item[3])
        # end if
        datetime_end = datetime.now()
        training_duration = (datetime_end - datetime_beg).total_seconds()
        if (the_label == "Category"):
            metrics = {'Accuracy': float(accuracy)}
            metrics['F1 score'] = float(f1_score_value)
            metrics['Precision'] = float(precision)
            metrics['Recall'] = float(recall)
            metrics['Cross entropy'] = float(cross_entropy_value)
            reference_value = float(cross_entropy_value)
        else:
            metrics = {'RMSE_loss': float(rmse_loss)}
            metrics['R^2 score'] = float(r2_score)
            reference_value = float(rmse_loss)
        # end if
        metrics['training_duration'] = float(training_duration)
        metrics['inference_latency'] = float(inference_latency)
        folder = 'neural_networks'
        save_model(folder, model, hyperparameters, metrics)
        print("reference_value, loss_function_baseline = ", reference_value, loss_function_baseline)
        if (reference_value < loss_function_baseline):
            folder = 'best_nn'
            save_model(folder, model, hyperparameters, metrics)
            if (the_label == "Category"):
                loss_function_baseline = cross_entropy_value
            else:
                loss_function_baseline = rmse_loss
            # end if
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
the_label = "Category"  #  "Price_Night" or "bedrooms" or "Category"
if (the_label == "Price_Night"):
    features_labels_tuple = load_airbnb("Price_Night")
elif (the_label == "Category"):
    features_labels_tuple = load_airbnb("Category")
elif (the_label == "bedrooms"):
    features_labels_tuple = load_airbnb("bedrooms")
# end if
df = features_labels_tuple[0]
X, y, X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(df)
#print(df.dtypes)
if __name__ == '__main__':
    # Load hyperparameters from file.
    config = get_nn_config()
    # Prepare the neural network dataset.
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_len = int(len(dataset)*0.7)
    val_len = int(len(dataset)*0.15)
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    # Prepare the neural network dataloaders.
    train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=16)
    val_loader = DataLoader(train_set, shuffle=False, batch_size=16)
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_len, len(train_set)-train_len])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    # Prepare the features and labels for the neural network model.
    example = next(iter(train_loader))
    features, labels = example
    # Run the code either as an individual test run or as a neural network
    run_type = "neural_network"   # "individual" or "neural_network"
    if (run_type == "individual"):
        # Use this for individual test runs; the following "find_best_nn" is 
        # for complex runs
        epochs = 10
        model = Neural_Network(config, 1, 1)
        grid = GridSearchCV(estimator=model, param_grid=config)
        grid_result = grid.fit(X_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, 
                                     grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    elif (run_type == "neural_network"):
        number_of_classes = 1
        if (the_label == "Category"):
            labels = pd.DataFrame()
            label_to_predict = df.iloc[:,:1]
            labels.insert(0, 'Category', label_to_predict)
            df = df.select_dtypes(include=['number'])
            unique_classes = labels['Category'].unique()
            number_of_classes = len(unique_classes)
        # end if
        # Instantiate neural network model and train it for a training set with 
        # a selection of hyperparameters. Save results of each run.
        find_best_nn(config, train_loader, the_label, number_of_classes)
    # end if
# end if

### end programme
