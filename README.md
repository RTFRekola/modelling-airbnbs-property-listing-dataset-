# Modelling Airbnb's Property Listing Dataset
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team. 

## Table of Contents
1. [Project Description](#section-1)
2. [Installation Instructions](#section-2)
3. [Usage Instructions](#section-3)
4. [File Structure](#section-4)
5. [License Information](#section-5)
6. [Project History](#section-6)

## Project Description

This is a project in which a typical selection of Airbnb accommodation data was used for creating a set of machine learning models and neural networks in order to make predictions based on the data. The project consisted of writing Python 3 code to perform the aforementioned tasks and displaying graphical and numeric data on the results. 

## Installation Instructions

When preparing to run this code for the first time, do the following:
- create a directory for the code; e.g. in Linux terminal <b>mkdir airbnb</b>
- copy files <b><i>neural_network.py</i></b>, <b><i>modelling.py</i></b>, <b><i>tabular_data.py</i></b>, <b><i>nn_config.yalm</i></b>, and <b><i>copyright.txt</i></b> into the dictionary you just created
- go to this directory and create the following directories (see also "file structure" below) in it; e.g. in Linux terminal <b>mkdir models models/classification models/classification/decision_tree models/classification/gradient_boosting models/classification/logistic_regression models/classification/random/forest models/regression models/regression/decision_tree models/regression/gradient_boosting models/regression/linear_regression models/regression/random_forest neural_networks neural_networks/best_nn neural_networks/regression</b>

## Usage Instructions

Open a terminal and go to the directory where you installed the code. If you want to test a series of regression and classification machine learning models, run <b>python3 modelling.py</b> and check the contents of the directory <b><i>models</i></b> for results. If you want to test neural networks, run <b>python3 neural_network.py</b> and check the contents of the directory <b><i>neural_networks</i></b> for results. Once run, the folders in <b><i>models/classification</i></b> and <b><i>models/regression</i></b> will have files in them. Similarly the there will be folders named with the day and time of the run in <b><i>neural_networks/regression</i></b>, which contain files with results of the run. The files in these folders contain the best model, the hyperparameters used in this case, and the loss or score of the run with these hyperparameters. 

## File Structure

You may choose to have a specific location and name for your project directory. We can assume the location is in home dictionary and that the directory for the project exists in it and is called <b>airbnb</b>. Therefore your file structure, in your home folder, would look like the following (folders are shown in <b><i>bold italics</i></b>):

<b><i>airbnb</i></b><br />
&nbsp;&nbsp;&nbsp; |_ nn_config.yaml<br />
&nbsp;&nbsp;&nbsp; |_ tabular_data.py<br />
&nbsp;&nbsp;&nbsp; |_ modelling.py<br />
&nbsp;&nbsp;&nbsp; |_ neural_networks.py<br />
&nbsp;&nbsp;&nbsp; |_ copyright.txt<br />
&nbsp;&nbsp;&nbsp; |_ <b><i>models</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; |_ <b><i>classification</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; |_ <b><i>decision_tree</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; |_ <b><i>gradient_boosting</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; |_ <b><i>logistic_regression</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; |_ <b><i>random_forest</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; |_ <b><i>regression</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; |_ <b><i>decision_tree</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; |_ <b><i>gradient_boosting</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; |_ <b><i>linear_regression</i></b><br />
&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; |_ <b><i>random_forest</i></b><br />
&nbsp;&nbsp;&nbsp; |_ <b><i>neural_networks</i></b><br />
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; |_ <b><i>best_nn</i></b><br />
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; |_ <b><i>regression</i></b><br />

## License Information

Copyright 2023, Rami Rekola

Copying and distribution of these files, with or without modification, are permitted in any medium without royalty, provided the copyright notice and this notice are preserved. These files are offered as-is, without any warranty.

## Project History

### Set Up the Environment
Created a new repository in GitHub called <b>modelling-airbnbs-property-listing-dataset</b> and added the URL for the remote repository where to push the local repository.

### Data Preparation
Load, clean and prepare input data for further work. 

Loaded in a tabular dataset as a large zip file. Unzipped the tabular dataset and read in the file <i>listing.csv</i> as a Pandas DataFrame. The tabular dataset contains these columns:

- ID: Unique identifier for the listing
- Category: The category of the listing
- Title: The title of the listing
- Description: The description of the listing
- Amenities: The available amenities of the listing
- Location: The location of the listing
- guests: The number of guests that can be accommodated in the listing
- beds: The number of available beds in the listing
- bathrooms: The number of bathrooms in the listing
- Price_Night: The price per night of the listing
- Cleanliness_rate: The cleanliness rating of the listing
- Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
- Location_rate: The rating of the location of the listing
- Check-in_rate: The rating of check-in process given by the host
- Value_rate: The rating of value given by the host
- amenities_count: The number of amenities in the listing
- url: The URL of the listing
- bedrooms: The number of bedrooms in the listing

Performed data cleaning by removing rows with missing ratings values, rows with missing descriptions, and unnecessary text in the beginning of each description. Further removed empty elements from descriptions and converted newline characters into single whitespaces. Merged list elements in descriptions into a single string. Replaced empty values with a "1" in guests, beds, bathrooms, and bedrooms. Saved the improved dataset into <i>clean_tabular_data.csv</i>.

Extracted all numerical values (columns with numbers) into a Pandas DataFrame without column headers. Created a tuple with this data frame as the first element and the list of column headers as the second element. 

### Regression Model Creation

Build a system to train and evaluate machine learning models. 

Read the numerical values data in as the data for machine learning models. Used four different sklearn regression models to analyse the Airbnb data with: SGDRegressor, DecisionTreeRegressor, RandomForestRegressor, and GradientBoostingRegressor. Analysed the quality of SGDRegressor with RMSE and R-squared methods, and a specific function that manually performs a grid search over a range of hyperparameter values in order to find the ones that provide highest performance metrics value. Furthermore, analysed each of the models with the sklearn built-in GridSearchCV. 

### Classification Model Creation

Imported Airbnb data with column "Category" as the label. Trained sklearn LogisticRegressor to predict the category from the data. Calculated the F1 score, precision score, recall score and accuracy score for both the training and test set. Performed grid search over a range of hyperparameters to find the best combination, with output consisting of the best model, the best hyperparameter values and the validation accuracy. Expanded the same to three more classifiers: DecisionTreeClassifier, RandomForestClassifier, and GradientBoostingClassifier. Implemented search for the best overall classification model. 

### Configurable Neural Network Creation

#### First model - price per night as the label

Initiated a PyTorch dataset with a tuple of <b><i>features</i></b> - the numerical values of the Airbnb data - and <b><i>label</i></b> - the price per night. Created a data shuffling dataloader for the train and test sets. Split the training set into train and validation sets. Defined a PyTorch model class for the neural network and a function to train the model. Had the training go through all of the data in batches and a number of epochs to optimise the model parameters. 

Set up TensorBoard to visualise the behaviour of the tests. 

Modified the code to read operational test parameters from a YAML file. These parameters include a set of optimisers, learning rates, hidden layer widths and model depths.
Added functionality to save each of the tests in a separate folder of its own. The save data includes the RMSE loss, R^2 score, model training time, and prediction making time. 

Ran through tests with a range of parameters and saved each in its own folder and the best separately in a specific folder reserved for the best model. The input parameters tested the neural networks with three different depths (2, 3 and 4) and three different widths (8, 12 and 16). For simplicity each of the hidden layers had the same width. Each of these variations was fed into three optimisers (SGD, Adagrad and Adam). Additionally three values of learning rate were tested (0.0001, 0.0002 and 0.0004). All the test values combined produced altogether 81 different variations. The neural network architecture schematics of the tested variations are shown graphically below. 

![modelling-airbnbs-property-listing-dataset-](img/NNarchitecture.png?raw=true "Neural network architecture for all tested variations with two, three or four hidden layers, each with a width of either 8, 12 or 16.")

Loss function behaviour in each of the tests was monitored with TensorBoard in VSC. The combined graphs of all 81 variations are shown below. 

![modelling-airbnbs-property-listing-dataset-](img/All81.png?raw=true "Loss functions of all 81 tests.")

To distinguish the quality of data between different optimisers, graphs containing only data from one of the three optimisers are given below (top: SGD, middle: Adagrad, bottom: Adam).

![modelling-airbnbs-property-listing-dataset-](img/SGD.png?raw=true "Loss functions of tests done with SGD optimiser.")
![modelling-airbnbs-property-listing-dataset-](img/Adagrad.png?raw=true "Loss functions of tests done with Adagrad optimiser.")
![modelling-airbnbs-property-listing-dataset-](img/Adam.png?raw=true "Loss functions of tests done with Adam optimiser.")

While there are no noticeable differences between optimisers, the 12 test results with best observable decreasing nature of the loss function, regardless of the optimiser, were chosen for the graph below. Even though some of them rise initially, they all descend nicely towards the end. 

![modelling-airbnbs-property-listing-dataset-](img/Best12.png?raw=true "Loss functions of the best 12 tests.")

Using the TensorBoard functionality of ignoring outliers in the chart scaling, the results (below) are easier to see. 

![modelling-airbnbs-property-listing-dataset-](img/Best12-excl_outliers.png?raw=true "Loss functions of the best 12 tests without outliers.")

Using the TensorBoard functionality of smoothing, at the maximum setting, the results (below) are yet easier to see.

![modelling-airbnbs-property-listing-dataset-](img/Best12-max_smooth.png?raw=true "Loss functions of the best 12 tests with maximum smoothing.")

The best prediction was produced with Adagrad, using two hidden layers, each with a width of eight, and learning rate of 0.0001. The schematic is given below.

![modelling-airbnbs-property-listing-dataset-](img/NN-2x8.png?raw=true "Network diagram for the best prediction, achieved at two hidden layers, both with a width of eight.")

The loss function graph from TensorBoard for the best prediction is given below.

![modelling-airbnbs-property-listing-dataset-](img/TheBest_Adagrad_8_2_1.png?raw=true "Loss function of the best prediction.")

It is worth noting that the best model parameters are not identifying a trend of best values - at least if compared to the set of 12 chosen best descending loss functions. While optimiser Adam was only present in 2 of the best 12 tests, the rest were roughly equally divided between SGD and Adagrad. The model depth was most commonly four, with only two depths of two in the set of 12. The hidden layer width and learning rate were both equally distributed between the three options of each.

#### Second model - number of bedrooms as the label

The same code as above was run again, but this time the label was changed from the price per night to the number of bedrooms. Again, each optimiser had good and not so good results for various parameters. Adagrad had fewer good ones, but that may be statistically insignificant - especially since the best test run was made with Adagrad. Unlike with the price per night results, there was a clear trend with the number of bedroom results favouring higher number of hidden layer widths and smaller number of hidden layers. Also higher learning rate values seemed to produce better results. All the 81 results are shown as a TensorBoard graph below. 

![modelling-airbnbs-property-listing-dataset-](img/b-All81.png?raw=true "Loss functions of all 81 tests.")

Again, the 12 cases with clearest decreasing nature of the loss function were selected for a graph of their own, below. 

![modelling-airbnbs-property-listing-dataset-](img/b-Best12.png?raw=true "Loss functions of the best 12 tests.")

In fact, results with the number of bedrooms are so much better than the results with the price per night that outliers were not excluded. Instead the graph for the best 12 cases was smoothed maximally in TensorBoard, below. 

![modelling-airbnbs-property-listing-dataset-](img/b-Best12_smooth.png?raw=true "Loss functions of the best 12 tests with maximum smoothing.")

The best prediction was produced with Adagrad, using four hidden layers, each with a width of twelve, and learning rate of 0.0002. 
The loss function graph from TensorBoard for the best prediction is given below.

![modelling-airbnbs-property-listing-dataset-](img/b-TheBest_Adagrad_12_4_2.png?raw=true "Loss function of the best prediction.")
