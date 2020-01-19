# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:10:48 2020

@author: AAAAAAAAAAAAAAAAAA
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



#instatiate label encoder object
le = preprocessing.LabelEncoder()

#read in data csv
data = pd.DataFrame(pd.read_csv("PastHires.csv"))

 
categorical_feature_mask = data.dtypes==object


categorical_cols = data.columns[categorical_feature_mask].tolist()

data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))


#take in data of hired
y = data['Hired']
#and the rest of the data 
x = data.drop('Hired',axis = 1)


#Train Test Split of 80/20, seed = 0
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.2, random_state = 0)

#gaussian Naive bayes
clf = GaussianNB()
clf.fit(xTrain,yTrain)

results = clf.predict(xTest)



for result in results:
    if result == 1:
        print("hired")
    else:
        print("not hired")
        
        
