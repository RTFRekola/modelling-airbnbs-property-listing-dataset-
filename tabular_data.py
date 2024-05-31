#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon 12 Jun 2023 at 18:36 UT
Last modified on Sat 13 Jan 2024 at 16:33 UT

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

import numpy as np
import pandas as pd


# ==============================================
# ===   Functions   ============================
# ==============================================

def clean_tabular_data(df):

    '''
    Load a csv file, fix the data, and save the clean data as a new file.

    Input and return variable: 
    - df = Pandas dataframe with the data
    '''

    def fix_shifted_rows(df):

        '''
        Move shifted data to correct place, or, rather, remove the bad row.

        Input and return variable: 
        - df = Pandas dataframe with the data
        '''

        # Remove the one problematic row.
        df = df.drop(labels=586, axis=0)
        return df
    # end fix_shifted_rows


    def remove_rows_with_missing_ratings(df):

        '''
        Remove rows with missing values in ratings columns.

        Input and return variable: 
        - df = Pandas dataframe with the data
        '''

        drop_list = ['Cleanliness_rating', 'Accuracy_rating',
                     'Communication_rating', 'Location_rating', 
                     'Check-in_rating', 'Value_rating']
        for item in drop_list:
            df = df.dropna(subset=[item])
        # end for
        return df
    # end remove_rows_with_missing_ratings


    def combine_description_strings(df):

        '''
        Remove rows with empty description. Remove 'About this space' prefixes. 
        Remove empty items and convert carriage returns into single whitespaces.
        Merge list items into the same string. 

        Input and return variable: 
        - df = Pandas dataframe with the data
        '''

        def make_corrections(description_in):
            description_in_progress_11 = description_in.replace("'About this space', ", "")
            description_in_progress_2 = description_in_progress_11.replace("', '", "")
            description_in_progress_3 = description_in_progress_2.replace('\\n', ' ')
            description_in_progress_4 = description_in_progress_3.strip()
            description_out = (description_in_progress_4.replace("['", "").
                               replace('["', '').replace('"]', '').
                               replace("']", ""))
            return description_out

        # make_corrections
        df = df.dropna(subset=['Description'])
        df['Description'] = df.apply(lambda row: 
                                     make_corrections(row['Description']), 
                                     axis=1)
        return df
    # end combine_description_strings


    def set_default_feature_values(df):

        '''
        Replace empty values with a unity in selected columns. 

        Input and return variable: 
        - df = Pandas dataframe with the data
        '''

        fill_list = ['guests', 'beds', 'bathrooms', 'bedrooms']
        for item in fill_list:
            df[item] = df[item].fillna(1)
        # end for
        return df
    # end set_default_feature_values

    # Fix shifted rows
    df = fix_shifted_rows(df)
    # Remove rows with missing rating values
    df = remove_rows_with_missing_ratings(df)
    # Merge list items into one string
    df = combine_description_strings(df)
    # Replace empty values with 1 in selected columns.
    df = set_default_feature_values(df)
    return df
# end clean_tabular_data


def load_airbnb(in_label):

    '''
    Return all numerical values, or features, as a pandas dataframe and their
    headers, or labels, as a list in the tuple format (features, labels)

    Input variable:
    - in_label = the database column to be used as the label

    Internal variables: 
    - df = Pandas dataframe with the data, read in from a file
    - df_selection = Pandas dataframe with selected columns of the original df
    - labels_list = list of dataframe column headers

    Return variable:
    - features_labels_tuple = tuple with selected dataframe data and column 
                              headers in it
    '''

    df = pd.read_csv("../airbnb-local/tabular_data/clean_tabular_data.csv")
    df = df.replace(['Treehouses', 'Category', 
                     'Chalets', 'Amazing pools', 'Offbeat', 
                     'Beachfront'], [1, 1, 2, 3, 4, 5])
    # Change column type of these columns into numbers
    label_price_night = "Price_Night"
    label_category = "Category"
    label_bedrooms = "bedrooms"

    if (in_label == "Price_Night"):
        label_2 = "Category" ; label_3 = "bedrooms"
    elif (in_label == "Category"):
        label_2 = "Price_Night" ; label_3 = "bedrooms"
    elif (in_label == "bedrooms"):
        label_2 = "Price_Night" ; label_3 = "Category"
    # end if
    df[[in_label, "guests", "beds", "bathrooms", "Cleanliness_rating", 
        "Accuracy_rating", "Communication_rating", "Location_rating", 
        "Check-in_rating", "Value_rating", "amenities_count", label_2, 
        label_3]] = df[[in_label, "guests", "beds", "bathrooms", 
                        "Cleanliness_rating", "Accuracy_rating", 
                        "Communication_rating", "Location_rating", 
                        "Check-in_rating", "Value_rating", 
                        "amenities_count", label_2, 
                        label_3]].apply(pd.to_numeric)
    labels_list = [in_label, "guests", "beds", "bathrooms", 
                   "Cleanliness_rating", "Accuracy_rating", 
                   "Communication_rating", "Location_rating", 
                   "Check-in_rating", "Value_rating", "amenities_count", 
                   label_2, label_3]
    '''
    if (in_label == "Price_Night"):
        df[["Price_Night", "guests", "beds", "bathrooms", "Cleanliness_rating", 
            "Accuracy_rating", "Communication_rating", "Location_rating", 
            "Check-in_rating", "Value_rating", "amenities_count", 
            "bedrooms"]] = df[["Price_Night", "guests", "beds", "bathrooms", 
                               "Cleanliness_rating", "Accuracy_rating", 
                               "Communication_rating", "Location_rating", 
                               "Check-in_rating", "Value_rating", 
                               "amenities_count", 
                               "bedrooms"]].apply(pd.to_numeric)
        labels_list = ["Price_Night", "guests", "beds", "bathrooms", 
                       "Cleanliness_rating", "Accuracy_rating", 
                       "Communication_rating", "Location_rating", 
                       "Check-in_rating", "Value_rating", "amenities_count", 
                       "bedrooms"]
        labels_list.append("Category")
        # labels_list.append("bedrooms")
    elif (in_label == "Category"):
        df[["Category", "guests", "beds", "bathrooms", "Cleanliness_rating", 
            "Accuracy_rating", "Communication_rating", "Location_rating", 
            "Check-in_rating", "Value_rating", "amenities_count", 
            "bedrooms"]] = df[["Category", "guests", "beds", "bathrooms", 
                               "Cleanliness_rating", "Accuracy_rating", 
                               "Communication_rating", "Location_rating", 
                               "Check-in_rating", "Value_rating", 
                               "amenities_count", 
                               "bedrooms"]].apply(pd.to_numeric)
        labels_list = ["Category", "guests", "beds", "bathrooms", 
                       "Cleanliness_rating", "Accuracy_rating", 
                       "Communication_rating", "Location_rating", 
                       "Check-in_rating", "Value_rating", "amenities_count", 
                       "bedrooms"]
        labels_list.append("Price_Night")
        # labels_list.append("bedrooms")
    elif (in_label == "bedrooms"):
        df[["bedrooms", "guests", "beds", "bathrooms", "Cleanliness_rating", 
            "Accuracy_rating", "Communication_rating", "Location_rating", 
            "Check-in_rating", "Value_rating", "amenities_count", 
            "Price_Night"]] = df[["bedrooms", "guests", "beds", "bathrooms", 
                                  "Cleanliness_rating", "Accuracy_rating", 
                                  "Communication_rating", "Location_rating", 
                                  "Check-in_rating", "Value_rating", 
                                  "amenities_count", 
                                  "Price_Night"]].apply(pd.to_numeric)
        labels_list = ["bedrooms", "guests", "beds", "bathrooms", 
                       "Cleanliness_rating", "Accuracy_rating", 
                       "Communication_rating", "Location_rating", 
                       "Check-in_rating", "Value_rating", "amenities_count", 
                       "Price_Night"]
        labels_list.append("Category")
        # labels_list.append("Price_Night")
    # end if
    '''
    df_selection = df[labels_list]
    df_selection.columns = ['', '', '', '', '', '', '', '', '', '', '', 
                            '', '']
    features_labels_tuple = (df_selection, labels_list)
    # end if        
    return features_labels_tuple
# end load_airbnb

if __name__ == "__main__":
    # Load data from a CSV file into a Pandas DataFrame
    df = pd.read_csv("../airbnb-local/tabular_data/listing.csv")
    # Clean the data
    df = clean_tabular_data()
    # Save processed data from a Pandas DataFrame into a CSV file
    df.to_csv("../airbnb-local/tabular_data/clean_tabular_data.csv", 
              encoding='utf-8', index=False)
# end if

# end programme
