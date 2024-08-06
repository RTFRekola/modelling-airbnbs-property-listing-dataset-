# Modelling Airbnb's Property Listing Dataset
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team. 

## Project Description

This is a project in which a typical selection of Airbnb accommodation data was used for creating a set of machine learning models and neural networks in order to make predictions based on the data. The project consisted of writing Python 3 code to perform the aforementioned tasks and displaying graphical and numeric data on the results. 

## Installation Instructions

When preparing to run this code for the first time, do the following:

- create a directory for the code; e.g. in Linux terminal give command 

```
mkdir airbnb
```

Now you can either (git clone the repository): 
```
git clone https://github.com/RTFRekola/modelling-airbnbs-property-listing-dataset-.git
```
If unfamiliar, see more information on [git cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). 

Or (perform the same manually):
- copy files <b><i>neural_network.py</i></b>, <b><i>modelling.py</i></b>, <b><i>tabular_data.py</i></b>, <b><i>nn_config.yalm</i></b>, and <b><i>copyright.txt</i></b> into the dictionary you just created
- go to this directory and create the following directories (see also "file structure" below) in it; e.g. in Linux terminal 

```
mkdir models models/classification models/classification/decision_tree models/classification/gradient_boosting models/classification/logistic_regression models/classification/random/forest models/regression models/regression/decision_tree models/regression/gradient_boosting models/regression/linear_regression models/regression/random_forest neural_networks neural_networks/best_nn neural_networks/regression
```

Finally you should update your Python environment to match the dependencies given in the file <b><i>requirements.txt</i></b>

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

See the [Project History in Wiki](https://github.com/RTFRekola/modelling-airbnbs-property-listing-dataset-/wiki). This includes outcomes of the models and their metrics, with some awesome graphics, too. 