# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:22:56 2022

@author: levir
"""


import numpy as np
import pandas as pd
#import pandas_profiling
import matplotlib.pyplot as plt
#import category_encoders as ce
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
from matplotlib import pyplot
import csv
import sys



def DataCleaner(file):

    dataset = pd.read_csv(file)
    dataset.head()
    filename = file.split("/")[-1].split(".")[0]
    
    print(dataset.isnull().values.any())
#    not always appear in the way we know - N / A, but in different forms like NA, na and more.
#Therefore, the solution I have found for this is to turn all these words, into the same word. To perform data clearing suitable for all blank fields.

    Missing_V = ["na", "N/a","NA", "",]
    dataset = pd.read_csv(file,na_values = Missing_V)
    print(dataset.head())
    
    null_values = dataset.isnull().sum()
    null_values = round((null_values/dataset.shape[0] * 100), 2)
    print(null_values.sort_values(ascending=False))
    #the columns that have more than sixty percent empty fields, are meaningless
    high_nan_rate_columns = null_values[null_values > 60].index
    
    print(high_nan_rate_columns)
    #To delete those columns I used the function
    dataset_cleaned = dataset.copy()
    dataset_cleaned.drop(high_nan_rate_columns, axis=1, inplace=True)
    
    null_values2 = dataset_cleaned.isnull().sum()
    null_values2 = round((null_values2/dataset_cleaned.shape[0] * 100), 2)
    print(null_values2.sort_values(ascending=False))
    
    ##### Numerical columns
    null_values_columns_dataset = dataset_cleaned.isnull().sum().sort_values(ascending=False)
    numerical_col_null_values = dataset_cleaned[null_values_columns_dataset.index].select_dtypes(include=['float64', 'int64']).columns
    # for each column
    for c in numerical_col_null_values:
    # Get the mean
        mean = dataset_cleaned[c].mean()
        # replace the NaN by mode
        dataset_cleaned[c].fillna(mean, inplace=True)
        
    ##### Categorical columns
    #categ_col_null_values = dataset_cleaned[null_values_columns_dataset.index].select_dtypes(include=['object',]).columns
    categ_col_null_values = dataset_cleaned[null_values_columns_dataset.index].select_dtypes(exclude=[np.number]).columns
    # for each column
    #categ_col_null_values.sort_values(ascending=False)
    for c in categ_col_null_values:
        # Get the most frequent value (mode)
        mode = dataset_cleaned[c].value_counts().index[0]
        # replace the NaN by mode
        dataset_cleaned[c].fillna(mode, inplace=True)
    
    print(dataset_cleaned.isnull().values.any())
    
    null_values = dataset_cleaned.isnull().sum()
    null_values = round((null_values/dataset_cleaned.shape[0] * 100), 2)
    null_values.sort_values(ascending=False)
    #avoid double columns I used the function
    dataset_cleaned = dataset_cleaned.drop_duplicates()
    #delete the colums with the sape value
    for col in dataset_cleaned.columns:
        if len(dataset_cleaned[col].unique()) == 1:
            dataset_cleaned.drop(col,inplace=True,axis=1)
    
    categ_values = dataset_cleaned.select_dtypes(include=['object']).columns
    x= dataset_cleaned._get_numeric_data()
    #fit and transform the data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_numeric_std = pd.DataFrame(x_scaled)
    dataset_cleaned = pd.merge(X_numeric_std,  dataset_cleaned[categ_values], left_index=True, right_index=True)
    
    dataset_cleaned.hist()
    pyplot.show()
    
    # 1. import
    x= dataset_cleaned._get_numeric_data()
    StandardScaler_scaler = preprocessing.StandardScaler()
    #fit and transform the data
    x_scaled = StandardScaler_scaler.fit_transform(x)
    X_numeric_std = pd.DataFrame(x_scaled)
    dataset_cleaned = pd.merge(X_numeric_std,  dataset_cleaned[categ_values], left_index=True, right_index=True)
    print(dataset_cleaned)
    
    dataset_cleaned.hist()
    pyplot.show()
    
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    #Create correlation matrix
    corr_matrix = dataset_cleaned.corr().abs()
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    dataset_cleaned.drop(to_drop, axis=1, inplace=True)
    
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    #Create correlation matrix
    corr_matrix = dataset_cleaned.corr().abs()
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    return dataset_cleaned


file =sys.argv[1]
filename = file.split("/")[-1].split(".")[0]
DataCleaner(file).to_csv(filename+'_cleaned.csv', sep='\t')



    
    
    
    
    
    
    
    
    
    