# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:40:06 2023

@author: santo
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("C:\datasets\ethnic diversity.csv")

df.dtypes
#type casting
#salaries are in float type we convert it into int
df.Salaries =df.Salaries.astype(int)
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
#scatter plot
df.plot(kind='scatter',x='Salaries',y='age')
#box plot
df.plot(kind='box',x='Salaries',y='age')

#identify the duplicates
#finding duplicates
df_new = pd.read_csv("C:\datasets\education.csv")
duplicate=df_new.duplicated()
#o/p of this function is in single column
#there is if any duplicate then o/p be True
#there is if any no-duplicate then o/p be False
duplicate
sum(duplicate)
#o/p will be 0
df_new1 = pd.read_csv("C:\datasets\mtcars_dup.csv")
duplicate1=df_new1.duplicated()

duplicate1
sum(duplicate1)


df_new2=df_new1.drop_duplicates()
df_new2.duplicated()

import pandas as pd
import seaborn as sns
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
#find outliers in salaries column
sns.boxplot(df.Salaries)
#find outliers in age column
sns.boxplot(df.age) #No outliers
#calculate outliers
#calculate IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#have observed IQR in variable explorer
#no,because IQR is in capital letters
#treated as constant
IQR
#but if we will try as I,Iqr,iqr then it is showing
I=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
Iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR #lower limit
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR #upper limit

#lower limit it is negative so make it 0
#___________________________________________________________________________________
#Trimming 
#___________________________________________________________________________________
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
#check outlier_df column in variable explorer
df.trimmed=df.loc[~outliers_df]
df.shape
df.trimmed.shape
#___________________________________________________________________________________

#replacement technique
#___________________________________________________________________________________
#drawback of trimming technique is losing the data
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
df.describe()
#record no.23 got outliers
#map all the outlier values to upper limit
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,                    True,False)))
#if the values are greater than upper_limit
#map it to upper limit ,less than lower limit
#map it to lower limit , if it is within range
sns.boxplot(df_replaced[0]);

#___________________________________________________________________________________
#Winsorizer
#___________________________________________________________________________________

import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
import seaborn as sns

df=pd.read_csv("C:\datasets\ethnic diversity.csv")
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Salaries']
                  )
#___________________________________________________________________________________
df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])

#___________________________________________________________________________________

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 08:41:52 2023

Preprocessing of data

"""
import pandas as pd
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
df.shape
df.columns
df.describe()
#check datatypes of dataframe
df.dtypes

#TYPECASTING
#changing salaries datatype in int
df.Salaries=df.Salaries.astype(int)
df.dtypes

#similarly age data changing to float it must be in float
#presently it is int
df.age=df.age.astype(float)
df.dtypes


#Identify the duplicates
df_new=pd.read_csv("C:\datasets\ethnic diversity.csv")
duplicate=df_new.duplicated()
#output of this function is single column
#if there is duplicate records output - True
duplicate
sum(duplicate)


df_new1=pd.read_csv("C:\datasets\ethnic diversity.csv")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)

#Dropping the duplicates
df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
sum(duplicate2)


#OUTLIER TREATMENT
import pandas as pd
import seaborn as sns
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
sns.boxplot(df.Salaries)
sns.boxplot(df.age)
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
IQR
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR



#Trimming outliers
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df]
df.shape
df_trimmed.shape
#in trimming process we loose our data
#so we do the mapping
#Replacement Technique / Mapping
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
df.describe()
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
#if the values are greater than upper limit map it to upper limit
##if the values are lower than lower limit map it to lower limit
#if it is within range,then keep it as it is
sns.boxplot(df_replaced[0])
#---------------------------------------------------------
# winsorizer
import pandas as pd
import numpy as np
import seaborn as sns
from feature_engine.outliers import Winsorizer

df=pd.read_csv("C:\datasets\ethnic diversity.csv")
winsor=Winsorizer(capping_method = 'iqr',
                  tail='both',
                  fold =1.5,
                  variables=['Salaries']
                  )
import seaborn as sns
df_t=winsor.fit_transform(df[['Salaries']])
#------------------------
sns.boxplot(df['Salaries'])
sns.boxplot(df_t["Salaries"])
#------------------------

df_bost =pd.read_csv("C:/2-datasets/Boston.csv.xls")
"""
Businesss Constraints

minimize: Minimize the house prices in boston area

Maximize:Increasing the profit

objective: predict the median value of owner.based on different owners.The price is 
predict by using neighbourhood houses.

Business Constraint: budget limitation and resourse constraint

Data dictionary:check the data will be quantitative or qualitativee,weather
the data is structured or unstructured
"""
df_bost
df_bost.describe()
df_bost.dtypes
df_bost.isnull().sum()
df_bost.head()

sns.pairplot(df_bost,hue='crim')

sns.countplot(x='age',data=df_bost)
sns.hisplot(data=df_bost,x="age")
df_bost.plot(kind ='scatter',x='crim',y='age')

#---------------------------------------------
# ZERO VARIANCE AND NEAR ZERO VARIANCE
#if there is no variance in the feature , then the ML model
# will not get any intelligence,so it is better to ignore those feature
 
import pandas as pd
df=pd.read_csv()
df.var()
# here EmpID and ZIP is nominal data
# salary has 4.441953e+08 is 444195  which is not close to zero
# similarly for age 
# boh the feature having cosiderable varience 
df.var()==0
'''EmpID       False
Zip         False
Salaries    False
age         False
dtype: bool
'''


# none of them are equal
df.var(axis=0)==0
#----------------------------
import pandas as pd
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
# check for null values
df.isna().sum()
"""
Position            43
State               35
Sex                 34
MaritalDesc         29
CitizenDesc         27
EmploymentStatus    32
Department          18
Salaries            32
age                 35
Race                25
dtype: int64
"""
import nunmpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
# check the dataframe
df['Salaries']=pd.DataFrame(mean_imputer.fit_transform(df[['salaries']]))
# check the dataframe
df['Salaries'].isna().sum()
#-----------------------------------------------------------
import pandas as pd
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
df.head(10)
df.info()
# it gives size,null values ,rows,column & column data
df.describe()

df['Salaries_new']=pd.cut(df['Salaries'],bins=[min(df.Salaries),df.Salaries.mean(),max(df.Salaries)],labels=["low","high"])

df.Salaries_new.value_counts()

df['Salaries']=pd.cut(df['Salaries'],bins=[min(df.Salaries),df.Salaries.quantile(0.25),df.Salaries.mean(),df.Salaries.quantile(0.75),max(df.Salaries)],labels=["group1","group2","group3","group4"])

df.Salaries_new.value_counts()
#-------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
df.shape
df.drop(['Index'],axis=1,inplace=True)
#check df again
df
df_new=pd.get_dummies(df)
df_new.shape
# here we get 30 rows & 14 cilumns
# we are getting two column for homly and gender,one column
#delete 2nd column of gender and 2nd column of homly

df_new.drop(['Gender_Male','Homly_Yes'],axis=1,inplace=True)
df_new.shape

#now we are getting 30,12

df_new.rename(columns={'Gender_Female':'Gender','Homly_no':'homly'})

#-------------------------------------------------------------
import pandas as pd 
import numpy as np
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
df
df.shape
df.drop(['EmpID'],axis=1,inplace=True)
df_new=pd.get_dummies(df)
df_new.shape

df_new.rename(columns={'Gender_Female':'Gender','Homly_no':'homly'})


#-------------------------------------------------------------
#one hot encoder
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
#we use ethnic diversity dataset
df=pd.read_csv("C:\datasets\ethnic diversity.csv")
df.columns
#we have salaries and age as numerical column,let us make them 
#at position 0 and 1 so to make further data preprocessing easy

df=df[['Salaries','age','Position','State','Sex','MaritalDesc','CitizenDesc','EmploymentStatus', 'Department','Race']]

#check the dataframe in variable explorer
#we want only nominal data and ordinal data for processing
#hence skipped 0th and first column and applied to one hot encoder
enc_df=pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray())
#
#________________

#label encoder
#________________

from sklearn.preprocessing import LabelEncoder
#creating instance of label encoder
labelencoder=LabelEncoder()
#split your data into input and output variables
X=df.iloc[:,0:9] #First eight columns for X and 9 th for y or  
y=df['Race']
df.columns
#we have nominal data sex,maritaldesc,citizendesc we want to convert to label encoder
X['Sex']=labelencoder.fit_transform(X['Sex'])
X['MaritalDesc']=labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc']=labelencoder.fit_transform(X['CitizenDesc'])
#label encoder y
y=labelencoder.fit_transform(y)
#This is going to create an array, hence convert 
#it back to dataframe
y=pd.DataFrame(y)
df_new=pd.concat([X,y],axis=1)
# if  you will see variables explporer, y do not have column name
# hence rename the column
df_new=df_new.rename(columns={0:'Race'})

#---------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()

#we use ethatic diversity dataset

df=pd.read_csv("C:/2-Dataset/ethnic diversity.csv")
df.columns

#we have salaries and age in the numericalcolumn
#at the position 0 and 1

df=df[['Salaries','age','Position','State','Sex','MaritalDesc','CitizenDesc']]
#check the dataframe in variable explore
#we want only nominal data and ordinal data for processing
#hence skipped 0 th and first column and applied to one hot  encoder

enc_df=pd.DataFrame(enc.fit_transform(df.iloc[:,2:]).toarray())
#label encoder

from sklearn.preprocessing import LabelEncoder
#creating instance of label encoder

labelencoder=LabelEncoder()

X=df.iloc[:,0:9]
y=df["Race"]
df.columns

#we have nominal data,sex,maritaldesc,citizendesc
#convert label to encoder

X['Sex']=labelencoder.fit_transform(X['MaritalDesc'])
X['MaritalDesc']=labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc']=labelencoder.fit_transform(X['CitizenDesc'])
#label encoder y

y=labelencoder.fit_transform(y)
#to create array hence convert
#back to the dataframe

y=pd.DataFrame(y)
df_new=pd.concat([X,y],axis=1)

df_new=df_new.rename(columns={0:'Race'})



############################

