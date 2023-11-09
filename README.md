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

## Installation Instructions

## Usage Instructions

## File Structure

## License Information

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
