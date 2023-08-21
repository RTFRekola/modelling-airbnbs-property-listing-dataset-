# Modelling Airbnb's Property Listing Dataset
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team. 

## Table of Contents
1. [Introduction](#introduction)
2. [Set Up the Environment](#section-1)
3. [Data Preparation](#section-2)

## Set Up the Environment
Created a new repository in GitHub called modelling-airbnbs-property-listing-dataset and added the URL for the remote repository where to push the local repository.

## Data Preparation
Load, clean and prepare input data for further work. 

Loaded in a tabular dataset as a large zip file. Unzipped it and read in listing.csv as a Pandas DataFrame. Removed rows with missing ratings values. Removed rows with missing descriptions. Removed unnecessary text in the beginning of each description. Removed empty elements from descriptions and converted newline characters into single whitespaces. Merged list elements in descriptions into a single string. Replaced empty values with a "1" in guests, beds, bathrooms, and bedrooms. Saved the improved dataset into clean_tabular_data.csv.

Extracted all numerical values (columns with numbers) into a Pandas DataFrame without column headers. Created a tuple with this data frame as the first element and the list of column headers as the second element. 
