#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon 12 Jun 2023 at 18:36 UT
Last modified on Mon 21 Aug 2023 at 20:44 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

import pandas as pd
import numpy as np


# ==============================================
# ===   Functions   ============================
# ==============================================

def clean_tabular_data(df):

    '''
    Load a csv file, fix the data, and save the clean data as a new file.
    '''

    def fix_shifted_rows(df):

        '''
        Move shifted data to correct place
        '''

        ''' The following does not work
        def shifter(row):
            return np.hstack((np.delete(np.array(row), [6]), [np.nan]))
        # end shifter

        df.iloc[19] = df.iloc[19].fillna(1)
        mask = df['url'] == '46'
        df.loc[mask, :] = df.loc[mask, :].apply(shifter, axis=1)
        df = df.drop([19], axis=1)
        '''
        ''' The following does not work either
        mask = df['url'] == '46'
        df.loc[mask, 6:18] = df.loc[mask, 7:19].shift(periods = -1, axis = 1)
        '''

        # Just removing row 588 instead, at least for now
        df = df.drop(labels=586, axis=0)
        return df
    # end fix_shifted_rows


    def remove_rows_with_missing_ratings(df):

        '''
        Remove rows with missing values in ratings columns.
        '''

        df = df.dropna(subset=['Cleanliness_rating'])
        df = df.dropna(subset=['Accuracy_rating'])
        df = df.dropna(subset=['Communication_rating'])
        df = df.dropna(subset=['Location_rating'])
        df = df.dropna(subset=['Check-in_rating'])
        df = df.dropna(subset=['Value_rating'])
        return df
    # end remove_rows_with_missing_ratings


    def combine_description_strings(df):

        '''
        Remove rows with empty description. Remove 'About this space' prefixes. 
        Remove empty items and convert carriage returns into single whitespaces.
        Merge list items into the same string. 
        '''

        def make_corrections(description_in):
            description_mid1 = description_in.replace("'About this space', ", 
                                                      "")
            description_mid2 = description_mid1.replace("', '", "")
            description_mid3 = description_mid2.replace('\\n', ' ')
            description_mid4 = description_mid3.strip()
            description_out = (description_mid4.replace("['", "").
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
        '''

        df['guests'] = df['guests'].fillna(1)
        df['beds'] = df['beds'].fillna(1)
        df['bathrooms'] = df['bathrooms'].fillna(1)
        df['bedrooms'] = df['bedrooms'].fillna(1)
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


def load_airbnb():

    '''
    Return all numerical values, or features, as a pandas dataframe and their
    headers, or labels, as a list in the tuple format (features, labels)
    '''

    if __name__ == "__main__":

        # Load data from a CSV file into a Pandas DataFrame
        df = pd.read_csv("../airbnb-local/tabular_data/listing.csv")

        # Clean the data
        df = clean_tabular_data()

        # Save processed data from a Pandas DataFrame into a CSV file
        df.to_csv("../airbnb-local/tabular_data/clean_tabular_data.csv", 
                  encoding='utf-8', index=False)
    else:
        df = pd.read_csv("../airbnb-local/tabular_data/clean_tabular_data.csv")

        df_selection = df[["guests", "beds", "bathrooms", "Price_Night", 
                           "Cleanliness_rating", "Accuracy_rating", 
                           "Communication_rating", "Location_rating", 
                           "Check-in_rating", "Value_rating", 
                           "amenities_count", "bedrooms"]]
        df_selection.columns = ['', '', '', '', '', '', '', '', '', '', '', '']

        # change column type of these columns into numbers
        df[["Price_Night", "guests", "beds", "bathrooms", "Cleanliness_rating", 
            "Accuracy_rating", "Communication_rating", "Location_rating", 
            "Check-in_rating", "Value_rating", "amenities_count", 
            "bedrooms"]] = df[["guests", "beds", "bathrooms", "Price_Night", 
                               "Cleanliness_rating", "Accuracy_rating", 
                               "Communication_rating", "Location_rating", 
                               "Check-in_rating", "Value_rating", 
                               "amenities_count", 
                               "bedrooms"]].apply(pd.to_numeric)

        labels_list = ["Price_Night", "guests", "beds", "bathrooms", 
                       "Cleanliness_rating", "Accuracy_rating", 
                       "Communication_rating", "Location_rating", 
                       "Check-in_rating", "Value_rating", 
                       "amenities_count", "bedrooms"]
        features_labels_tuple = (df_selection, labels_list)
    # end if        
    return features_labels_tuple
# end load_airbnb

# end programme
