#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon 12 Jun 2023 at 18:36 UT
Last modified on Mon 12 Jun 2023 at 18:36 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

import pandas as pd


# ==============================================
# ===   Functions   ============================
# ==============================================

def clean_tabular_data():

    def remove_rows_with_missing_ratings(df):

        '''
        Remove rows with missing values in ratings columns.
        '''

        df = df.dropna(subset=['Cleanliness_rate'])
        df = df.dropna(subset=['Accuracy_rate'])
        df = df.dropna(subset=['Location_rate'])
        df = df.dropna(subset=['Check-in_rate'])
        df = df.dropna(subset=['Value_rate'])
        return df
    # end remove_rows_with_missing_ratings


    def combine_description_strings(df):

        '''
        Remove rows with empty description. Remove 'About this space' prefixes. 
        Remove empty items and convert carriage returns into single whitespaces.
        Merge list items into the same string. 
        '''

        def remove_empty_items_and_join(description_list):
            description_list = [x for x in description_list if x]
            " ".join(description_list)
            return description_list
        # end remove_empty_items_and_join

        df = df.dropna(subset=['Description'])
        df['Description'] = df['Description'].replace(r'About this space',
                                                      '', regex=True)
        df['Description'] = df.apply(lambda row: remove_empty_items_and_join(row['Description']), axis=1)
        df['Description'] = df['Description'].replace(r'\n',' ', regex=True)
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


    # ==============================================
    # ===   Main content of clean_tabular_data   ===
    # ==============================================

    if __name__ == "__main__":

        # Load data from a CSV file into a Pandas DataFrame
        df = pd.read_csv("../airbnb-local/tabular_data/listing.csv")

        # Remove rows with missing rating values
        df = remove_rows_with_missing_ratings(df)

        # Merge list items into one string
        df = combine_description_strings(df)

        # Replace empty values with 1 in selected columns.
        df = set_default_feature_values(df)

        # Save processed data from a Pandas DataFrame into a CSV file
        df.to_csv("../airbnb-local/tabular_data/clean_tabular_data.csv", 
                  encoding='utf-8', index=False)
    # end if

    # end clean_tabular_data
