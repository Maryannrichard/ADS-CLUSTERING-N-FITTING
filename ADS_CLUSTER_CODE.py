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
file_path = 'WORLD_COUNTRIES.xlsx'
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

# extracting data for 2020.
df_2020 = df2[2020]
# Convert the Series to DataFrame using unstack()
df_2020_unstack = df_2020.unstack()

# Select columns for the year 2020
series = ['Individuals using the Internet (% of population)', 
                       'GDP per capita (current US$)',
                       'Access to electricity (% of population)',
                       'ICT goods imports (% total goods imports)',
                       'Employment to population ratio, 15+, total (%) (national estimate)'
                      ]
df_2020_sub = df_2020_unstack[series]

# Calculate correlation matrix for the cleaned subset
df_2020_corr = df_2020_sub.corr()

# Display the correlation matrix for the subset
print("Correlation Matrix for the Year 2020:")
print(df_2020_corr)

# Plot the correlation matrix
sns.heatmap(df_2020_corr, annot=True, cmap='Greys', fmt=".2f")

# Set plot properties
plt.title('Correlation Matrix Heatmap for the Year 2020', fontweight='bold')
plt.show()

# Individuals using internet and GDP looks good for clustering
# create data frame for the two selected series
gdp_col = df_2020.xs('GDP per capita (current US$)', level='Series Name', 
                     axis=0, drop_level=True)
internet_col =df_2020.xs('Individuals using the Internet (% of population)', 
                         level='Series Name', 
                         axis=0, drop_level=True)
# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(internet_col,gdp_col,s=100, alpha=0.7, color='blue', 
            edgecolors='black', marker='*')

# Set the scatter plot label
plt.xlabel('Individuals using the Internet (% of population)')
plt.ylabel('GDP per capita (current US$)')
plt.title('Scatter Plot on original data: Internet users vs. GDP',
          fontweight='bold')


# Prepare data for cluster
df_cluster = pd.DataFrame({'Individuals using the Internet (% of population)':
                           internet_col.values,
    'GDP per capita (current US$)': gdp_col.values})

# print the cluster dataframe
print(df_cluster)

# create a scaler object for use in scaling the data for the cluster
scaler = pp.RobustScaler()

#set up the scaler
scaler.fit(df_cluster)

# apply the scaling to normalize the data
df_norm = scaler.transform(df_cluster)
print(df_norm)

plt.figure(figsize=(8, 8))

plt.scatter(df_norm[:,0], df_norm[:, 1], s=100, alpha=0.7, color='red', 
            edgecolors='white', marker='*')

plt.xlabel('Individuals using the Internet (% of population)')
plt.ylabel('GDP per capita (current US$)')
plt.title('Scatter plot normalized data on internet users vs. GDP', 
          fontweight='bold')

# calculate silhouette score to known the number of clusters to use
for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

# performing  the clustering
plt.figure(figsize=(10.0, 10.0))

ncluster = 3 # number of clusters

# set up clusterer with number of cluster
kmeans = cl.KMeans(n_clusters=ncluster, n_init=20, random_state=40)

# Doing the clustering. The result will be stored in clusters
kmeans.fit(df_norm)

# Extracting the labels, i.e. the cluster nmumber
labels = kmeans.labels_
'''Extract the cluster centres. These are the cluster centres for the
normalised data. To show them with the original data they need to
be rescaled'''
centres = kmeans.cluster_centers_
centres = scaler.inverse_transform(centres)
# centres is a list of x and y values. Extract x and y.
xcen = centres[:, 0]
ycen = centres[:, 1]

# extract x and y values of data points
x = internet_col
y = gdp_col

plt.scatter(internet_col, gdp_col, 40, labels, marker="*",cmap='Set1',
            label='countries cluster')
# For the centres only one colour is selected
plt.scatter(xcen, ycen, 50, "k", marker="d", label='cluster groups')
plt.xlabel("Individual using internet")
plt.ylabel("GDP per capita")
plt.title('Cluster visual: Internet users vs GDP',fontweight='bold')
plt.legend()

# adding the cluster memebership information to the dataframe
df_cluster["labels"] = labels

# Preparing data to use for fit.
# selected 2 countries from each cluster
# group 0 cluster countries
country = 'Canada'
series_name = 'Individuals using the Internet (% of population)'
df_canada = series_data(df2, country, series_name)

country = 'United States'
series_name = 'Individuals using the Internet (% of population)'
df_usa = series_data(df2, country, series_name)

# group 1 cluster countries
country = 'South Africa'
series_name = 'Individuals using the Internet (% of population)'
df_SA = series_data(df2, country, series_name)

country = 'Mexico'
series_name = 'Individuals using the Internet (% of population)'
df_mexico = series_data(df2, country, series_name)

# group 2 cluster countries
country = 'France'
series_name = 'Individuals using the Internet (% of population)'
df_France = series_data(df2, country, series_name)

country = 'Spain'
series_name = 'Individuals using the Internet (% of population)'
df_spain = series_data(df2, country, series_name)

# define the curve fit for India and South Africa
param, covar = opt.curve_fit(
    exponential, df_usa["Year"], 
    df_usa["Individuals using the Internet (% of population)"],
    p0=(1.2e12, 0.03))
df_usa['fit'] = exponential(df_usa['Year'], *param)

param1, covar1 = opt.curve_fit(
    exponential, df_SA["Year"], 
    df_SA["Individuals using the Internet (% of population)"],
    p0=(1.2e12, 0.03))
df_SA['fit'] = exponential(df_SA['Year'], *param1)

# plotting the fit graph for USA
plt.figure()

df_usa.plot('Year',["Individuals using the Internet (% of population)",
                      'fit'])
plt.title("Fitted graph for Internet users(% of population for USA", 
          fontweight='bold')
plt.xlabel("Year")

# Set x-axis to display only integer years
plt.xticks(df_usa["Year"].astype(int)[::3])

# plotting teh fit graphfor South Africa
plt.figure()

df_SA.plot('Year',["Individuals using the Internet (% of population)",
                      'fit'])
plt.title("Fitted graph for Internet users(% of population for South Africa", 
          fontweight='bold')
plt.xlabel("Year")

# Set x-axis to display only integer years
plt.xticks(df_SA["Year"].astype(int)[::3])


# forcast for USA and South Africa for the next 10 years
# create array for forecasting
year = np.linspace(2001, 2030, 100)
forecast = exponential(year, *param)
forecast1= exponential(year,*param1)

# calculating sigma values using errors function
sigma = err.error_prop(year, exponential, param, covar)
sigma1 = err.error_prop(year, exponential, param1, covar1)

#finding the limits for confidence intervals
up = forecast + sigma
up1 = forecast1 + sigma1
low = forecast - sigma
low1 = forecast1 - sigma1

#plot the forecast graph forUSA
plt.figure()
plt.plot(df_usa["Year"], df_usa[
    "Individuals using the Internet (% of population)"], 
    label="Original data")
plt.plot(year, forecast, label="forecast data")
plt.fill_between(year, low, up, color="yellow", alpha=0.7,
                 label='confidence interval')
plt.xlabel("year")
plt.ylabel("Individuals using the Internet (% of population)")
plt.title('10years forecast on internet users in USA', fontweight='bold')
plt.legend()

# plot the forecast graph for South Africa
plt.figure()
plt.plot(df_SA["Year"], df_SA[
    "Individuals using the Internet (% of population)"], 
    label="Original data")
plt.plot(year, forecast1, label="forecast data")
plt.fill_between(year, low1, up1, color='blue',alpha=0.7,
                 label='confidence interval')
plt.xlabel("year")
plt.ylabel("Individuals using the Internet (% of population)")
plt.title('10years forecast on internet users in South Africa', 
          fontweight='bold')
plt.legend()

plt.show()

# Analysing countries in the same cluster to understand trend
selected_countries = ['Canada', 'United States', 'Denmark', 'Austria']

# selecting series name
selected_series = ['Individuals using the Internet (% of population)', 
                   'Access to electricity (% of population)', 
                   'ICT goods imports (% total goods imports)',
                  'Employment to population ratio, 15+, total (%) (national estimate)']

# Filter df2 for the selected countries and series
df_selected = df2.loc[(selected_countries, selected_series), :]

# Reset the index for better manipulation
df_selected = df_selected.reset_index()

# Melt the DataFrame to reshape it for seaborn barplot
df_melted = pd.melt(df_selected, id_vars=['Country Name', 'Series Name'], 
                    var_name='Year', value_name='Value')

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Country Name', y='Value', hue='Series Name', data=df_melted)

# Set plot title and labels
plt.title('Bar Chart for Selected Countries in the same cluster')
plt.xlabel('Country')
plt.ylabel('Value')

# Move legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()

# trend plot on countries from different clusters GDP
selected_countries2 = ['Canada', 'United States', 'South Africa', 
                      'Mexico', 'France', 'Spain']
# Filter df2 for the selected countries and 'Individuals using the Internet 
df_selected2 = df2.loc[(selected_countries2, 
                        'Individuals using the Internet (% of population)'), :]

# Transpose the DataFrame for easier plotting
df_selected2_T = df_selected2.T

# Plot the trend for the selected countries
ax = df_selected2_T.plot(marker='o', figsize=(12, 8))

# Get the handles and labels for the plot
handles, labels = ax.get_legend_handles_labels()

# Create a new legend with only country names
new_labels = [label.split(',')[0] for label in labels]
ax.legend(handles, new_labels, title='Country')

# Set plot title and labels
plt.title('GDP- Selected Countries  from different clusters')
plt.xlabel('Year')
plt.ylabel('GDP')

# Set x-axis ticks to display only integer years
plt.xticks(df_selected2_T.index.astype(int)[::3])
# Show the plot
plt.show()