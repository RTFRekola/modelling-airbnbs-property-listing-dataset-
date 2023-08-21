#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu 6 Jul 2023 at 19:50 UT
Last modified on Thu 6 Jul 2023 at 19:50 UT 

@author: Rami T. F. Rekola 

Modelling Airbnb's Property Listing Dataset
===========================================
'''

from tabular_data import load_airbnb
import pandas as pd
import numpy as np
from sklearn import datasets, model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# ==============================================
# ===   Functions   ============================
# ==============================================


# ==============================================
# ===   Main programme   =======================
# ==============================================

features_labels_tuple = load_airbnb()

df = features_labels_tuple[0]
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X = scale(X)
y = scale(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

sgdr = SGDRegressor()
print(sgdr)

sgdr.fit(X_train, y_train)

score = sgdr.score(X_train, y_train)
print("R-squared:", score)
cv_score = cross_val_score(sgdr, X, y, cv = 10)
print("CV mean score: ", cv_score.mean())

