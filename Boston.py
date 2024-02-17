# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 08:20:33 2023

@author: santo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/datasets/Boston.csv")
df.dtypes
#type casting
#salaries are in float type we convert it into int
df.crim =df.crim.astype(int)
df.dtypes
#age are in int type we convert it into float
df.age=df.age.astype(float)
df.dtypes

#shape
df.shape
#columns
df.columns
#describe
df.describe()
#is null
df.isnull().sum()
#head
df.head()
#info
df.info()
#pairplt
sns.pairplot(df)
#
sns.pairplot(df,hue='age')
#countplt
sns.countplot(x='age',data=df)
sns.histplot(data=df,x='age')

sns.histplot(data=df,x='age',kde=True,stat='density') #pdf
sns.histplot(data=df,x='age',kde=True,stat='probability') #cdf


#scatter plot
df.plot(kind='scatter',x='crim',y='age')
#box plot
df.plot(kind='box',x='crim',y='age')
df.plot(kind='box',x='zn',y='age')
df.plot(kind='box',x='indus',y='age')
df.plot(kind='box',x='chas',y='age')
df.plot(kind='box',x='nox',y='age')
df.plot(kind='box',x='rm',y='age')
df.plot(kind='box',x='age',y='age')
df.plot(kind='box',x='dis',y='age')
df.plot(kind='box',x='rad',y='age')
df.plot(kind='box',x='tax',y='age')
df.plot(kind='box',x='ptratio',y='age')
df.plot(kind='box',x='black',y='age')
df.plot(kind='box',x='lstat',y='age')
df.plot(kind='box',x='medv',y='age')































