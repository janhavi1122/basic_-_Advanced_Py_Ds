# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:27:21 2023

@author: supremcourt
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("C:\datasets\mtcars.csv")
df.describe()
a=df.describe()
# initialize the scalar
scalar=StandardScaler()
df1=scalar.fit_transform(df)
dataset=pd.DataFrame(df)
res=dataset.describe()
# there if you will check re. in variable environment then you will
##########################################################3
df=pd.read_csv("C:\datasets\mtcars.csv")
df.columns
df.drop(['Employee_Name','EmpID','Zip'],axis=1,inplace=True)
a1=df.describe()
df=pd.get_dummies(df,drop_first=True)
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(df)
df1=df_norm.describe()

