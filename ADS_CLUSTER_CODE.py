# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:03:26 2024

@author: A
"""
#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import sklearn.preprocessing as pp
import sklearn.cluster as cl
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err

import warnings
warnings.filterwarnings("ignore")

# setting up functions for use.
#function 1' to be used in reading the data.
def data(file_path, columns_remove=None, index_col=None):
    '''
    This function reads a dataframe and returns 2 dataframes
    with one being the original data and the second being the 
    transposed off the first data having  multi-index columns

    Parameters:
    ------------------
    file_path : The name of the file to be read
    columns_remove: Deletes unwanted columns
    index_col : setting some columns as index'''
    
    # Read the data from the file
    df = pd.read_excel(file_path)

    # Clean the data by removing specified columns
    if columns_remove:
        df = df.drop(columns=columns_remove, errors='ignore')

    # Set the specified column as the index
    if index_col:
        df = df.set_index(index_col, drop=True)

    # Transpose the cleaned DataFrame
    df_T = df.T

    return df, df_T

# function 2' to be used in selecting data for fitting

def series_data(df2, country, series_name):
    # Select data for the specified country and series
    select_data = df2.loc[(country, series_name), :]

    # Reset the index and rename columns
    select_data = select_data.reset_index()
    select_data.columns = ['Year', f'{series_name}']

    return select_data

# function 3. to be used in checking the silhoutte score for use in
# selecting number of clusters to consider.

def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """

    # set up the clusterer with the number of expected clusters
    kmeans = cl.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score

# function 4. to be use to create an exponential model for fitting

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and 
    growth rate g."""
    
    # makes it easier to get a guess for initial parameters
    t = t - 2001
    f = n0 * np.exp(g*t)
    
    return f

# Read in the data
# defining the augments for the function to take
file_path = 'World_countries4.xlsx'
columns_remove =['Country Code','Series Code']
index_col = ['Country Name', 'Series Name']

df,df_T = data(file_path, columns_remove, index_col)

# Print original and transposed data
print("Original Data:")
print(df)

# Print transposed data
print("\nTransposed Data:")
print(df_T)

# cleaning the data
df.replace('...', pd.NA, inplace=True)

# convert data type to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Filter countries with any missing value in any series
df2 = df.groupby('Country Name').filter(lambda x: not x.isna().any().any())

# Reset the index
df2.reset_index(inplace=True)

# Set the index back to 'Country Name' and 'Series Name'
df2.set_index(['Country Name', 'Series Name'], inplace=True)

# Check the number of unique countries in the original DataFrame df
df_countries = df.index.get_level_values('Country Name').nunique()
print(f"Number of unique countries in the original DataFrame:{df_countries}")

# check the number of unique countries in the filtered DataFrame df2
df2_countries = df2.index.get_level_values('Country Name').nunique()
print(f"Number of unique countries in the filtered DataFrame:{df2_countries}")

'''# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   # allow for minus signs

# defining the augments for the function to take
file_path = 'World_countries4.xlsx'
columns_remove =['Country Code','Series Code']
index_col = ['Country Name', 'Series Name']

df,df_T = data(file_path, columns_remove, index_col)

# Print original and transposed data
print("Original Data:")
print(df)

# Print transposed data
print("\nTransposed Data:")
print(df_T)'''