# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 08:35:32 2023

@author: santo
"""
#Data Dictionary
#Index
#Speed: The car running by speed
#dist: The distance covered

from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("C:/datasets/Q1_a.csv")
#___________________________________________________________________________________
print(df.shape)
#output =(50, 3) 3 col, 50 rows
#___________________________________________________________________________________
#(Q) What are the column names in our dataset?
print (df.columns)
#Index(['Index', 'speed', 'dist'], dtype='object')
#___________________________________________________________________________________
print(df.dtypes)
#Index    int64
#speed    int64
#dist     int64
#dtype: object
#___________________________________________________________________________________
#scatter plot
df.plot(kind='scatter',x='speed',y='dist')
plt.show()
#___________________________________________________________________________________
#for checking datasate is balanced or not
df["speed"].value_counts()
#___________________________________________________________________________________
############################ HISTOGRAM###################################################


import seaborn as sns
plt.hist(df.speed)
sns.displot(x='speed',kde=True,bins=6,data=df)
#right skiwed
#___________________________________________________________________________________

import seaborn as sns
plt.hist(df.dist)
sns.displot(x='dist',kde=True,bins=6,data=df)
#left skiwed
#___________________________________________________________________________________

sns.FacetGrid(df, hue="speed", height=5) \
   .map(sns.distplot, "dist") \
   .add_legend();
plt.show();

#___________________________________________________________________________________
############################BAR PLOT###################################################
counts, bin_edges = np.histogram(df['speed'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(df['speed'], bins=10, density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);

plt.show();
#[0.04 0.06 0.08 0.12 0.16 0.1  0.14 0.16 0.02 0.12]
#[ 4.   6.1  8.2 10.3 12.4 14.5 16.6 18.7 20.8 22.9 25. ]
# we have virtually see that what parcentage of speed, dist have
# less than same values. 
#___________________________________________________________________________________
#boxplot for showing outlier
sns.boxplot(df.speed)
sns.boxplot(df.dist)
#___________________________________________________________________________________
#1st moment business decision
print("Means:")
print(np.mean(df["speed"]))
print(np.mean(df["dist"]))
#___________________________________________________________________________________

df.describe()
#Out[68]: 
#          Index      speed        dist
#count  50.00000  50.000000   50.000000
#mean   25.50000  15.400000   42.980000
#std    14.57738   5.287644   25.769377
#min     1.00000   4.000000    2.000000
#25%    13.25000  12.000000   26.000000
#50%    25.50000  15.000000   36.000000
#75%    37.75000  19.000000   56.000000
#max    50.00000  25.000000  120.000000
#___________________________________________________________________________________
#2nd moment business decision
print("\nStd-dev:");
print(np.std(df["speed"]))
print(np.std(df["dist"]))
#___________________________________________________________________________________
#skewness of speed
sp_lst=df['speed'].tolist()
print('skewness of speed',skew(sp_lst))
dist=df["dist"].tolist()
print('skewness of dist',skew(dist))
print(skew(dist,axis=0,bias=True))
print(kurtosis(dist,axis=0,bias=True))
#skewness of speed -0.11395477012828319
#skewness of dist 0.7824835173114966
#0.7824835173114966
#0.24801865717051808
#___________________________________________________________________________________







































