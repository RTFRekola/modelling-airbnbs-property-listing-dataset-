# Modelling Airbnb's Property Listing Dataset
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team. 

## Table of Contents
1. [Project Description](#project-description)
2. [Installation Instructions](#installation-instructions)
3. [Usage Instructions](#usage-instructions)
4. [File Structure](#file-structure)
5. [License Information](#license-information)
6. [Project History](#project-history)

## Project Description

This is a project in which a typical selection of Airbnb accommodation data was used for creating a set of machine learning models and neural networks in order to make predictions based on the data. The project consisted of writing Python 3 code to perform the aforementioned tasks and displaying graphical and numeric data on the results. 

## Installation Instructions

When preparing to run this code for the first time, do the following:
- create a directory for the code; e.g. in Linux terminal <b>mkdir airbnb</b>
- copy files <b><i>neural_network.py</i></b>, <b><i>modelling.py</i></b>, <b><i>tabular_data.py</i></b>, <b><i>nn_config.yalm</i></b>, and <b><i>copyright.txt</i></b> into the dictionary you just created
- go to this directory and create the following directories (see also "file structure" below) in it; e.g. in Linux terminal 

```
mkdir models models/classification models/classification/decision_tree models/classification/gradient_boosting models/classification/logistic_regression models/classification/random/forest models/regression models/regression/decision_tree models/regression/gradient_boosting models/regression/linear_regression models/regression/random_forest neural_networks neural_networks/best_nn neural_networks/regression
```

- update your Python environment to match the dependencies given in the file <b><i>requirements.txt</i></b>

## Usage Instructions

Open a terminal and go to the directory where you installed the code. If you want to test a series of regression and classification machine learning models, run 

```
python3 modelling.py
```

and check the contents of the directory <b><i>models</i></b> for results. If you want to test neural networks, run 

```
python3 neural_network.py
```

and check the contents of the directory <b><i>neural_networks</i></b> for results. Once run, the folders in <b><i>models/classification</i></b> and <b><i>models/regression</i></b> will have files in them. Similarly the there will be folders named with the day and time of the run in <b><i>neural_networks/regression</i></b>, which contain files with results of the run. The files in these folders contain the best model, the hyperparameters used in this case, and the loss or score of the run with these hyperparameters. 

## File Structure

You may choose to have a specific location and name for your project directory. We can assume the location is in home dictionary and that the directory for the project exists in it and is called <b>airbnb</b>. Therefore your file structure, in your home folder, would look like the following (folders are shown in <b><i>bold italics</i></b>):

<b><i>airbnb</i></b><br />
&nbsp;&nbsp;&nbsp; |_ README.md<br />
&nbsp;&nbsp;&nbsp; |_ nn_config.yaml<br />
&nbsp;&nbsp;&nbsp; |_ tabular_data.py<br />
&nbsp;&nbsp;&nbsp; |_ modelling.py<br />
&nbsp;&nbsp;&nbsp; |_ neural_networks.py<br />
&nbsp;&nbsp;&nbsp; |_ requirements.txt<br />
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

Best models and the hyperparameters used to get them (values rounded to 4 decimal places):

| Estimator                   | Training RMSE          | Validation RMSE         | Test RMSE              | Training R^2 | Validation R^2 | Test R^2 | 
| ---------                   | -------------          | ---------------         | ---------              | ------------ | -------------- | -------- | 
| SGD Regressor               | 0.1476×10<sup>15</sup> | 0.1497×10<sup>15</sup>  | 0.1432×10<sup>15</sup> | -1.2743      | -1.2587        | -1.3848  | 
| Decision Tree Regressor     | 1.2056                 | 1.4176                  | 1.40852                | -4.0183      | -6.5151        | -9.2523  | 
| Random Forest Regressor     | 0.4774                 | 1.3892                  | 1.3669                 | 0.75129      | -4.0843        | -4.0121  | 
| Gradient Boosting Regressor | 0.9816                 | 1.3784                  | 1.3510                 | -2.0177      | -7.3480        | -9.4427  | 

| Estimator                   | Hyperparameters for the best model results given above |
| ---------                   | ------------------------------------------------------ |
| SGD Regressor               | "alpha": 0.001, "eta0": 0.01, "learning_rate": "constant", "max_iter": 400, "power_t": 0.9 |
| Decision Tree Regressor     | "criterion": "poisson", "max_depth": 2, "min_samples_leaf": 1, "min_samples_split": 2 |
| Random Forest Regressor     | "bootstrap": true, "criterion": "friedman_mse", "min_samples_leaf": 1, "n_estimators": 50 |
| Gradient Boosting Regressor | "learning_rate": 0.2, "loss": "huber", "min_samples_leaf": 4, "n_estimators": 10 |

Negative R squared values indicate a non-linear x on y dependance or possibly some other problem, which is likely to be tied together with the fact that the number of data points used in this study was very limited to begin with. Whether overfitting could truly explain the high RMSE values of SGD Regressor is a matter of further study and might still not be resolved, given the small amount of data. 

### Classification Model Creation

Imported Airbnb data with column "Category" as the label. Trained sklearn LogisticRegressor to predict the category from the data. Calculated the F1 score, precision score, recall score and accuracy score for both the training and test set. Performed grid search over a range of hyperparameters to find the best combination, with output consisting of the best model, the best hyperparameter values and the validation accuracy. Expanded the same to three more classifiers: DecisionTreeClassifier, RandomForestClassifier, and GradientBoostingClassifier. Implemented search for the best overall classification model. 

Best models and the hyperparameters used to get them (values rounded to 4 decimal places):

| Estimator | Validation score | F1 score | Accuracy | Hyperparameters |
| ----- | ---------------- | -------- | -------- | ----------------| 
| Logistic Regression | 0.7626 | 0.8394 | 0.8370 | "C": 1, "max_iter": 10, "solver": "newton-cg" |
| Decision Tree Classifier | 0.7486 | 0.9677 | 0.9658 | "criterion": "gini", "max_depth": 20, "min_samples_leaf": 1, "min_samples_split": 4 |
| Random Forest Classifier | 0.8029 | 0.9079 | 0.9034 | "bootstrap": true, "criterion": "gini", "min_samples_leaf": 4, "n_estimators": 200 |
| Gradient Boosting Classifier | 0.7848 | 0.9779 | 0.9779 | "criterion": "squared_error", "learning_rate": 0.1, "min_samples_leaf": 1, "n_estimators": 50 |

Both classification and regression models could potentially be improved by going through even more hyperparameters and adding more values in each to test. Overall, a better model could be found by testing also more models, besides the ones tested here. 

### Configurable Neural Network Creation

Initiated a PyTorch dataset with a tuple of <b><i>features</i></b> - the numerical values of the Airbnb data - and <b><i>label</i></b>. Created a data shuffling dataloader for the train and test sets. Split the training set into train and validation sets. Defined a PyTorch model class for the neural network and a function to train the model. The model, the data loader and the number of epochs were fed into the training function, which then iterates through every batch in the dataset for the given number of epochs and optimises the model parameters. 

Set up TensorBoard to visualise the behaviour of the tests. 

Modified the code to read operational test parameters from a YAML file (see the image below). 

![modelling-airbnbs-property-listing-dataset-](img/nn_config_yaml.png?raw=true "Configuration file.")

These parameters include a set of optimisers, learning rates, hidden layer widths and model depths.
Added functionality to save each of the tests in a separate folder of its own. The save data includes the RMSE loss, R^2 score, model training time, and average prediction making time. 

Ran through tests with a range of parameters and saved each in its own folder and the best separately in a specific folder reserved for the best model. The input parameters tested the neural networks with three different depths (1, 3 and 6) and three different widths (4, 12 and 24). For simplicity each of the hidden layers had the same width. Each of these variations was fed into three optimisers (SGD, Adagrad and Adam). Additionally three values of learning rate were tested (0.0001, 0.001 and 0.01). All the test values combined produced altogether 81 different variations. The neural network architecture schematics of the tested variations are shown graphically below. 

![modelling-airbnbs-property-listing-dataset-](img/NNarchitecture.png?raw=true "Neural network architecture for all tested variations with one, three or six hidden layers, each with a width of either 4, 12 or 24.")

#### First model - price per night as the label

The test setting described above was first run with Price_Night as the label. Loss function behaviour in each of the tests was monitored with TensorBoard in VSC. The combined graphs of ten representative variations are shown below. TensorBoard smoothing was set to value 0.95.

![modelling-airbnbs-property-listing-dataset-](img/TB-Price_Night.png?raw=true "Loss functions of the best 15 tests.")

The best prediction was produced with Adagrad, using three hidden layers, each with a width of 24, and learning rate of 0.01. See the image below for the neural network architecture, which produced the best results. 

![modelling-airbnbs-property-listing-dataset-](img/Price_Night-3x24.png?raw=true "Best neural network for 'Price_Night'.")

#### Second model - number of bedrooms as the label

The same code as above was run again, but this time the label was changed from the price per night to the number of bedrooms. A group of ten representative results are shown again as a TensorBoard graph below. 

![modelling-airbnbs-property-listing-dataset-](img/TB-bedrooms.png?raw=true "Loss functions of the best 15 tests.")

The best prediction was produced with Adam, using three hidden layers, each with a width of 12, and learning rate of 0.01. See the image below for the neural network architecture, which produced the best results. 

![modelling-airbnbs-property-listing-dataset-](img/bedrooms-3x12.png?raw=true "Best neural network for 'bedrooms'.")

#### Third model - category as the label

The same code as above was run one more time. This time the label was set to the category. A group of ten representative results results are shown again as a TensorBoard graph below. 

![modelling-airbnbs-property-listing-dataset-](img/TB-Category.png?raw=true "Loss functions of the best 15 tests.")

The best prediction was produced with Adam, using 1 hidden layer, with a width of 4, and learning rate of 0.01. See the image below for the neural network architecture, which produced the best results. 

![modelling-airbnbs-property-listing-dataset-](img/Category-1x4.png?raw=true "Best neural network for 'Category'.")

Best optimisers and the hyperparameters used to get the best results in each of the three aforementioned cases (values rounded to 4 significant decimal places):

| Label | Optimiser | RMSE | R^2 score | Hyperparameters |
| ----- | --------- | ---- | --------- | ----------------| 
| Price_Night | Adam | 108.5084 | 0.2280 | "hidden_layer_width": 24, "model_depth": 3, "learning_rate": 0.01 |
| bedrooms | Adam | 0.4745 | 0.7725 | "hidden_layer_width": 12, "model_depth": 3, "learning_rate": 0.01 |
| Category | Adam | 1.313 | 0.0484 | "hidden_layer_width": 4, "model_depth": 1, "learning_rate": 0.01 |

The low values of the R^2 score are a result of the high variation, but - as seen in the graphs above - the smoothed curves show that the overall evolution of the values finds a fairly constant level, which is at a reasonable value considering the input data. Although Adam came up as the best optimiser in each case, the individual results for each run show that for otherwise identical test parameters, Adagrad was often very close to the results obtained with Adam. In general SGD performed less well, but a small number of results were close to the best obtained with Adam and Adagrad. While a choise of six hidden layers may seem excessive, and certainly six layers were not present in any of the best results, they produced results nearly as good as those with three hidden layers. Results obtained with only one hidden layer were often comparable with those of more layers, but generally always a little worse. For the price per night results, the hidden layer width of 24 produced clearly the best results. Otherwise there was not significant difference between layer widths of 4, 12 and 24. Best results for each of the three tested labels were obtained with learning rate of 0.01, but results with learning rate of 0.001 were better for some combinations of parameters and in a small number of cases learning rate of 0.0001 was better than either 0.01 or 0.001, but never better than both of them. 